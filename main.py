import os, json, pickle, logging, io, csv
from pathlib import Path
from datetime import datetime
from typing import Optional

# PostgreSQL (optional — graceful fallback auf JSONL wenn nicht verfügbar)
DATABASE_URL = os.getenv("DATABASE_URL")
_pg_available = False
try:
    import psycopg2
    import psycopg2.extras
    _pg_available = bool(DATABASE_URL)
except ImportError:
    pass

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from apscheduler.schedulers.background import BackgroundScheduler

# ── Setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Master-Bot ML Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR   = Path(os.getenv("DATA_DIR", "./data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

MIN_TRADES   = int(os.getenv("MIN_TRADES", "60"))    # Mindest-Trades für Training (erhöht für bessere Statistik)
PREDICT_CONF = float(os.getenv("PREDICT_CONF", "0.62"))  # Mindest-Konfidenz für vollen Trade (erhöht)

STRATEGIEN = ["mittel", "aggressiv", "smart", "konservativ", "optimiert", "test", "adaptive", "steady"]

# ── State ─────────────────────────────────────────────
models: dict = {}        # { strategie: Pipeline }
model_meta: dict = {}    # { strategie: { trainiert_am, accuracy, n_trades } }
trades_cache: dict = {}  # { strategie: [trades] }

# ── Daten laden ───────────────────────────────────────
def lade_trades_von_db(strategie: str) -> list:
    """Lädt Features direkt aus PostgreSQL (geteilt mit master-bot)"""
    if not _pg_available:
        return []
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT ts, strategie, side, equity::float, hour, weekday,
                   recent_wr5::float    AS "recentWR5",
                   recent_wr15::float   AS "recentWR15",
                   konsek,
                   rrr::float,
                   ausgefuehrt,
                   grund,
                   market_modus         AS "marketModus",
                   sl_dist_pct::float   AS "slDistPct",
                   reward_pct::float    AS "rewardPct",
                   spread::float,
                   session_london       AS "sessionLondon",
                   session_overlap      AS "sessionOverlap",
                   drawdown_pct::float  AS "drawdownPct",
                   pnl::float
            FROM features
            WHERE strategie=%s AND ausgefuehrt=true AND pnl IS NOT NULL
            ORDER BY ts ASC LIMIT 2000
        """, (strategie,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        if rows:
            log.info(f"[{strategie}] {len(rows)} Features aus PostgreSQL geladen")
            return rows
    except Exception as e:
        log.warning(f"[{strategie}] DB-Fehler, fallback auf JSONL: {e}")
    return []


def lade_trades(strategie: str) -> list:
    """Lädt Trades — bevorzugt PostgreSQL, Fallback auf features.jsonl / trades.json"""

    # 1. PostgreSQL (Railway shared DB — master-bot schreibt, ml-service liest)
    db_trades = lade_trades_von_db(strategie)
    if db_trades:
        trades_cache[strategie] = db_trades
        return db_trades

    # 2. Fallback: features.jsonl (lokal / Docker)
    features_file = DATA_DIR / "features.jsonl"
    trades_file   = DATA_DIR / "trades.json"
    trades = []

    if features_file.exists():
        with open(features_file) as f:
            for line in f:
                try:
                    t = json.loads(line.strip())
                    if t.get("strategie") == strategie and t.get("ausgefuehrt"):
                        trades.append(t)
                except:
                    pass

    # 3. Letzter Fallback: trades.json
    if not trades and trades_file.exists():
        with open(trades_file) as f:
            all_trades = json.load(f)
        raw = all_trades.get(strategie, [])
        for t in raw:
            trades.append({
                "pnl":         t.get("pnl", 0),
                "side":        t.get("side", "BUY"),
                "equity":      t.get("equity", 1000),
                "hour":        datetime.fromisoformat(t["datum"]).hour if "datum" in t else 12,
                "weekday":     datetime.fromisoformat(t["datum"]).weekday() if "datum" in t else 0,
                "ausgefuehrt": True,
            })

    trades_cache[strategie] = trades
    return trades

def baue_features(trades: list) -> tuple:
    """Baut Feature-Matrix und Labels aus Trade-Liste"""
    rows = []
    for i, t in enumerate(trades):
        pnl = t.get("pnl", 0)
        if pnl == 0:
            continue  # unklares Ergebnis überspringen

        # Rolling Metriken berechnen (letzte 5/15 Trades vor diesem)
        fenster5  = [x for x in trades[max(0,i-5):i]  if x.get("pnl", 0) != 0]
        fenster15 = [x for x in trades[max(0,i-15):i] if x.get("pnl", 0) != 0]

        wr5  = sum(1 for x in fenster5  if x["pnl"] > 0) / len(fenster5)  if fenster5  else 0.5
        wr15 = sum(1 for x in fenster15 if x["pnl"] > 0) / len(fenster15) if fenster15 else 0.5

        konsek = 0
        for j in range(i-1, -1, -1):
            if trades[j].get("pnl", 0) < 0:
                konsek += 1
            else:
                break

        hour = t.get("hour", 12)
        row = {
            # Zeitbasierte Features
            "hour":           hour,
            "weekday":        t.get("weekday", 0),
            "sessionLondon":  t.get("sessionLondon",  1 if 8  <= hour < 12 else 0),
            "sessionOverlap": t.get("sessionOverlap", 1 if 13 <= hour < 17 else 0),
            # Trade-Richtung
            "side_buy":       1 if t.get("side", "BUY") == "BUY" else 0,
            # Performance-Context
            "wr5":            round(wr5, 3),
            "wr15":           round(wr15, 3),
            "konsek":         min(konsek, 10),
            # Markt-Struktur (neu — aus Fix 3 in server.js)
            "rrr":            t.get("rrr") or 2.0,
            "slDistPct":      t.get("slDistPct") or 0.5,    # SL-Abstand in % vom Entry
            "rewardPct":      t.get("rewardPct") or 1.0,    # TP-Abstand in % vom Entry
            "spread":         t.get("spread") or 0.0,        # Bid-Ask Spread
            "drawdownPct":    t.get("drawdownPct") or 0.0,  # Drawdown vom Start-Equity
            "label":          1 if pnl > 0 else 0,
        }
        rows.append(row)

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    X = df.drop("label", axis=1).values
    y = df["label"].values
    return X, y

# ── Training ──────────────────────────────────────────
def trainiere(strategie: str) -> dict:
    log.info(f"[{strategie}] Starte Training...")
    trades = lade_trades(strategie)

    if len(trades) < MIN_TRADES:
        return {"status": "zu_wenig_daten", "n_trades": len(trades), "benoetigt": MIN_TRADES}

    X, y = baue_features(trades)
    if X is None or len(X) < MIN_TRADES:
        return {"status": "feature_fehler", "n_trades": len(trades)}

    # Klassen-Balance für XGBoost berechnen (kompensiert ungleiche Win/Loss-Verteilung)
    neg = int(np.sum(y == 0))
    pos = int(np.sum(y == 1))
    scale_pos = round(neg / pos, 3) if pos > 0 else 1.0

    # Pipeline: Skalierung + XGBoost (bessere Performance bei kleinen Datensätzen als Random Forest)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ))
    ])

    # Cross-Validation
    try:
        cv_scores = cross_val_score(pipeline, X, y, cv=min(5, len(X)//10 or 2), scoring="accuracy")
        accuracy  = round(float(cv_scores.mean()), 3)
    except:
        accuracy  = 0.0

    # Volles Training
    pipeline.fit(X, y)
    models[strategie] = pipeline

    meta = {
        "trainiert_am":  datetime.now().isoformat(),
        "n_trades":      len(trades),
        "n_features":    X.shape[0],
        "accuracy_cv":   accuracy,
        "klassen":       list(map(int, np.unique(y))),
        "win_rate_data": round(float(y.mean() * 100), 1),
    }
    model_meta[strategie] = meta

    # Modell speichern
    model_path = MODELS_DIR / f"{strategie}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "meta": meta}, f)

    log.info(f"[{strategie}] Training fertig — Accuracy: {accuracy:.1%}, Trades: {len(trades)}")
    return {"status": "ok", **meta}

def lade_gespeicherte_modelle():
    """Lädt gespeicherte Modelle beim Start"""
    for s in STRATEGIEN:
        model_path = MODELS_DIR / f"{s}.pkl"
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                models[s]     = data["pipeline"]
                model_meta[s] = data["meta"]
                log.info(f"[{s}] Modell geladen (Accuracy: {data['meta'].get('accuracy_cv', '?')})")
            except Exception as e:
                log.warning(f"[{s}] Modell konnte nicht geladen werden: {e}")

def retrain_alle():
    """Wöchentliches Auto-Retrain"""
    log.info("Auto-Retrain aller Strategien...")
    for s in STRATEGIEN:
        try:
            trainiere(s)
        except Exception as e:
            log.error(f"[{s}] Retrain-Fehler: {e}")

# ── Pydantic Models ───────────────────────────────────
class TrainRequest(BaseModel):
    strategie: Optional[str] = None  # None = alle trainieren

class PredictRequest(BaseModel):
    strategie: str
    side: str                         # "BUY" oder "SELL"
    equity: float
    hour: Optional[int] = None        # wird automatisch gesetzt wenn fehlt
    weekday: Optional[int] = None
    recentWR5: Optional[float] = None
    recentWR15: Optional[float] = None
    konsek: Optional[int] = None
    rrr: Optional[float] = 2.0
    threshold: Optional[float] = None  # Dynamischer Schwellwert vom master-bot (Market Mode)

# ── API Endpoints ─────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":    "ok",
        "modelle":   list(models.keys()),
        "db":        "postgresql" if _pg_available else "jsonl-fallback",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
def status():
    result = {}
    for s in STRATEGIEN:
        trades = trades_cache.get(s) or lade_trades(s)
        result[s] = {
            "trainiert":    s in models,
            "n_trades":     len(trades),
            "bereit":       len(trades) >= MIN_TRADES,
            "meta":         model_meta.get(s),
        }
    return result

@app.post("/train")
def train(req: TrainRequest):
    strategien = [req.strategie] if req.strategie else STRATEGIEN
    ergebnisse = {}
    for s in strategien:
        if s not in STRATEGIEN:
            ergebnisse[s] = {"status": "unbekannte_strategie"}
            continue
        try:
            ergebnisse[s] = trainiere(s)
        except Exception as e:
            ergebnisse[s] = {"status": "fehler", "detail": str(e)}
    return ergebnisse

@app.post("/predict")
def predict(req: PredictRequest):
    if req.strategie not in STRATEGIEN:
        raise HTTPException(400, "Unbekannte Strategie")

    # Kein Modell → Trade erlauben (fail-safe)
    if req.strategie not in models:
        trades = lade_trades(req.strategie)
        return {
            "empfehlung": "trade",
            "konfidenz":  None,
            "grund":      f"Kein Modell — {len(trades)}/{MIN_TRADES} Trades gesammelt",
            "trainiert":  False,
        }

    now = datetime.now()
    # Aktuell bekannte Win-Rates aus Cache berechnen
    trades = trades_cache.get(req.strategie, [])
    fenster5  = [t for t in trades[-5:]  if t.get("pnl", 0) != 0]
    fenster15 = [t for t in trades[-15:] if t.get("pnl", 0) != 0]
    wr5  = sum(1 for t in fenster5  if t.get("pnl",0) > 0) / len(fenster5)  if fenster5  else 0.5
    wr15 = sum(1 for t in fenster15 if t.get("pnl",0) > 0) / len(fenster15) if fenster15 else 0.5

    konsek = req.konsek if req.konsek is not None else 0
    X = np.array([[
        req.hour    if req.hour    is not None else now.hour,
        req.weekday if req.weekday is not None else now.weekday(),
        1 if req.side == "BUY" else 0,
        req.recentWR5  / 100 if req.recentWR5  is not None else wr5,
        req.recentWR15 / 100 if req.recentWR15 is not None else wr15,
        min(konsek, 10),
        req.equity,
        req.rrr or 2.0,
        req.recentWR5  if req.recentWR5  is not None else round(wr5 * 100, 1),
        req.recentWR15 if req.recentWR15 is not None else round(wr15 * 100, 1),
    ]])

    pipeline = models[req.strategie]
    proba    = pipeline.predict_proba(X)[0]
    win_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
    konfidenz = round(win_prob, 3)

    # Dynamischer Schwellwert: vom master-bot übergeben (Market Mode) oder Fallback auf PREDICT_CONF
    dyn_threshold    = float(req.threshold) if req.threshold is not None else PREDICT_CONF
    dyn_threshold    = max(0.50, min(0.95, dyn_threshold))  # Sicherheitsgrenzen

    # Konfidenz-basiertes Sizing statt binärem trade/skip
    # Hohes Vertrauen → volle oder erhöhte Größe; mittleres → halbe Größe; niedrig → skip
    SCHWELLE_REDUZIERT = 0.54   # unter Threshold aber über dieser → halbe Größe
    SCHWELLE_GROSS     = 0.72   # sehr hohe Konfidenz → 1.5x Größe

    if win_prob >= SCHWELLE_GROSS:
        empfehlung    = "trade"
        sizing_faktor = 1.5
        sizing_grund  = f"Sehr hohe Konfidenz ({win_prob:.0%}) → 1.5× Größe"
    elif win_prob >= dyn_threshold:
        empfehlung    = "trade"
        sizing_faktor = 1.0
        sizing_grund  = f"Konfidenz OK ({win_prob:.0%} ≥ {dyn_threshold:.0%})"
    elif win_prob >= SCHWELLE_REDUZIERT:
        empfehlung    = "reduziert"
        sizing_faktor = 0.5
        sizing_grund  = f"Mittlere Konfidenz ({win_prob:.0%}) → halbe Größe"
    else:
        empfehlung    = "skip"
        sizing_faktor = 0.0
        sizing_grund  = f"Zu niedrige Konfidenz ({win_prob:.0%} < {SCHWELLE_REDUZIERT:.0%})"

    grund = f"ML: {win_prob:.0%} Gewinn-Wahrscheinlichkeit — {sizing_grund}"

    log.info(f"[{req.strategie}] Predict: {req.side} → {empfehlung} ({win_prob:.0%}, threshold={dyn_threshold:.0%}, sizing={sizing_faktor}×)")
    return {
        "empfehlung":    empfehlung,
        "konfidenz":     konfidenz,
        "win_prob":      round(win_prob * 100, 1),
        "schwelle":      round(dyn_threshold * 100, 1),
        "sizing_faktor": sizing_faktor,
        "grund":         grund,
        "trainiert":     True,
    }

@app.get("/feature-importance/{strategie}")
def feature_importance(strategie: str):
    if strategie not in models:
        raise HTTPException(404, "Kein Modell für diese Strategie")
    rf = models[strategie].named_steps["model"]
    names = ["hour","weekday","sessionLondon","sessionOverlap","side_buy","wr5","wr15","konsek","rrr","slDistPct","rewardPct","spread","drawdownPct"]
    importances = rf.feature_importances_ if hasattr(rf, "feature_importances_") else []
    importance = sorted(zip(names[:len(importances)], importances), key=lambda x: -x[1])
    return {"strategie": strategie, "importance": [{"feature": n, "wert": round(float(v),4)} for n,v in importance]}

# ── TradingView CSV Import ────────────────────────────
def parse_tv_csv(content: str, strategie: str, start_equity: float = 1000.0) -> list:
    """
    Parst TradingView Strategy Tester CSV.
    Unterstützte Formate:
    - "Trade #,Type,Signal,Date/Time,Price,Contracts,Profit,Profit %,..."
    - "Datum;Typ;Preis;Gewinn;..." (DE-Format)
    """
    lines = content.strip().splitlines()
    if not lines:
        return []

    # Trennzeichen erkennen
    sep = ";" if lines[0].count(";") > lines[0].count(",") else ","
    reader = csv.DictReader(lines, delimiter=sep)

    # Spaltennamen normalisieren (TV ändert manchmal die Namen)
    COL_MAP = {
        # EN
        "type": "type", "signal": "type",
        "date/time": "datetime", "date": "datetime",
        "price": "price",
        "profit": "profit", "profit usd": "profit",
        "profit %": "profit_pct",
        "contracts": "contracts", "qty": "contracts",
        # DE
        "typ": "type", "signal": "type",
        "datum/uhrzeit": "datetime", "datum": "datetime",
        "kurs": "price", "preis": "price",
        "gewinn": "profit", "gewinn usd": "profit",
        "gewinn %": "profit_pct",
        "kontrakte": "contracts",
    }

    trades = []
    equity = start_equity
    entry_dt = None
    entry_price = None
    entry_side = None
    laufende_wr5  = []
    laufende_wr15 = []

    for raw_row in reader:
        # Spaltennamen normalisieren
        row = {}
        for k, v in raw_row.items():
            if k:
                norm = COL_MAP.get(k.strip().lower(), k.strip().lower())
                row[norm] = v.strip() if v else ""

        typ = row.get("type", "").lower()

        # Entry-Zeilen erkennen
        is_entry = any(x in typ for x in ["entry", "open", "long einstieg", "short einstieg", "kauf", "verkauf"])
        is_exit  = any(x in typ for x in ["exit", "close", "long ausstieg", "short ausstieg"])

        if is_entry:
            # Side bestimmen
            entry_side = "BUY" if any(x in typ for x in ["long", "buy", "kauf"]) else "SELL"
            try:
                entry_price = float(row.get("price", "0").replace(",", "."))
            except:
                entry_price = 0
            try:
                raw_dt = row.get("datetime", "")
                # Verschiedene Datumsformate
                for fmt in ["%Y-%m-%d %H:%M", "%d.%m.%Y %H:%M", "%Y-%m-%dT%H:%M", "%d/%m/%Y %H:%M"]:
                    try:
                        entry_dt = datetime.strptime(raw_dt, fmt)
                        break
                    except:
                        entry_dt = None
            except:
                entry_dt = datetime.now()

        elif is_exit and entry_side:
            # PnL aus Profit-Spalte
            profit_str = row.get("profit", "0").replace(",", ".").replace("%", "").strip()
            try:
                pnl = float(profit_str)
            except:
                pnl = 0

            gewinn = pnl > 0
            equity = round(equity + pnl, 2)

            # Rolling Win Rates berechnen
            laufende_wr5.append(1 if gewinn else 0)
            laufende_wr15.append(1 if gewinn else 0)
            if len(laufende_wr5)  > 5:  laufende_wr5.pop(0)
            if len(laufende_wr15) > 15: laufende_wr15.pop(0)

            wr5  = round(sum(laufende_wr5)  / len(laufende_wr5)  * 100, 1) if laufende_wr5  else 50.0
            wr15 = round(sum(laufende_wr15) / len(laufende_wr15) * 100, 1) if laufende_wr15 else 50.0

            konsek = 0
            for v in reversed(laufende_wr15):
                if v == 0: konsek += 1
                else: break

            feature = {
                "ts":          int(entry_dt.timestamp() * 1000) if entry_dt else 0,
                "datum":       entry_dt.isoformat() if entry_dt else "",
                "strategie":   strategie,
                "side":        entry_side,
                "pnl":         pnl,
                "equity":      equity,
                "hour":        entry_dt.hour if entry_dt else 12,
                "weekday":     entry_dt.weekday() if entry_dt else 0,
                "recentWR5":   wr5,
                "recentWR15":  wr15,
                "konsek":      konsek,
                "rrr":         2.0,   # TV liefert kein RRR direkt
                "ausgefuehrt": True,
                "quelle":      "tradingview_backtest",
            }
            trades.append(feature)

            # Reset für nächsten Trade
            entry_side = entry_price = entry_dt = None

    return trades

@app.post("/import-csv")
async def import_csv(
    file: UploadFile = File(...),
    strategie: str = Form(...),
    start_equity: float = Form(1000.0),
    retrain: bool = Form(True),
):
    if strategie not in STRATEGIEN:
        raise HTTPException(400, f"Unbekannte Strategie: {strategie}")

    content = (await file.read()).decode("utf-8", errors="replace")
    trades = parse_tv_csv(content, strategie, start_equity)

    if not trades:
        raise HTTPException(400, "Keine Trades gefunden — prüfe das CSV-Format")

    # In features.jsonl schreiben
    features_file = DATA_DIR / "features.jsonl"
    existing = set()
    if features_file.exists():
        with open(features_file) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    if t.get("strategie") == strategie and t.get("quelle") == "tradingview_backtest":
                        existing.add(t.get("ts", 0))
                except:
                    pass

    neu = [t for t in trades if t["ts"] not in existing]

    with open(features_file, "a") as f:
        for t in neu:
            f.write(json.dumps(t) + "\n")

    log.info(f"[{strategie}] CSV Import: {len(trades)} Trades gesamt, {len(neu)} neu")

    # Cache invalidieren
    if strategie in trades_cache:
        del trades_cache[strategie]

    ergebnis = {
        "status":         "ok",
        "strategie":      strategie,
        "trades_gesamt":  len(trades),
        "trades_neu":     len(neu),
        "trades_doppelt": len(trades) - len(neu),
    }

    if retrain and len(lade_trades(strategie)) >= MIN_TRADES:
        train_result = trainiere(strategie)
        ergebnis["training"] = train_result

    return ergebnis

def parse_telegram_export(content: str) -> list:
    """
    Parst Telegram Desktop JSON-Export.
    Format: { "messages": [{ "date": "...", "text": "..." }] }

    Erkennt Nachrichten wie:
    🟢 LONG — mittel\nSize: 2.5 | SL: 1920.0 | TP: 1960.0
    📝 [mittel] PnL +12.50€
    🔴 SHORT — aggressiv\nSize: 1.0 | SL: 1960.0 | TP: 1900.0
    """
    import re

    data = json.loads(content)
    messages = data.get("messages", data.get("chats", {}).get("list", [{}])[0].get("messages", []))

    # Alle Textnachrichten extrahieren
    def get_text(msg):
        t = msg.get("text", "")
        if isinstance(t, list):
            return "".join(p if isinstance(p, str) else p.get("text", "") for p in t)
        return str(t)

    # Regex-Muster
    RE_TRADE  = re.compile(r"(🟢|🔴).*(LONG|SHORT).*?—\s*(\w+)", re.IGNORECASE)
    RE_SIZE   = re.compile(r"Size:\s*([\d.]+)")
    RE_SL     = re.compile(r"SL:\s*([\d.]+)")
    RE_TP     = re.compile(r"TP:\s*([\d.]+)")
    RE_PNL    = re.compile(r"\[(\w+)\]\s+PnL\s+([+-]?[\d.]+)€?")
    RE_PNL2   = re.compile(r"PnL.*?([+-]?[\d.]+)€")  # Fallback
    RE_REGIME = re.compile(r"\[(\w+)\]\s+Regime:\s+(\w+)\s+→\s+(\w+)")

    trades_raw = {}   # { msg_id: { strategie, side, dt, sl, tp, size } }
    pnl_map   = {}    # { strategie: [pnl, ...] } — für Matching
    result    = []

    pending = {}  # { strategie: { side, dt, sl, tp, size } } — offene Trades

    for msg in messages:
        text = get_text(msg)
        if not text:
            continue

        raw_date = msg.get("date", "")
        try:
            dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
        except:
            dt = datetime.now()

        # Trade-Einstieg erkennen
        m_trade = RE_TRADE.search(text)
        if m_trade:
            emoji    = m_trade.group(1)
            side_str = m_trade.group(2).upper()
            strategie = m_trade.group(3).lower()
            side = "BUY" if side_str == "LONG" else "SELL"

            sl_m  = RE_SL.search(text)
            tp_m  = RE_TP.search(text)
            sz_m  = RE_SIZE.search(text)

            pending[strategie] = {
                "side":      side,
                "dt":        dt,
                "sl":        float(sl_m.group(1))  if sl_m  else 0,
                "tp":        float(tp_m.group(1))  if tp_m  else 0,
                "size":      float(sz_m.group(1))  if sz_m  else 1,
                "strategie": strategie,
            }
            continue

        # PnL-Nachricht erkennen
        m_pnl = RE_PNL.search(text)
        if not m_pnl:
            m_pnl2 = RE_PNL2.search(text)
            if m_pnl2:
                # Welche Strategie? — aus pending nehmen
                for strat, p in list(pending.items()):
                    pnl = float(m_pnl2.group(1))
                    entry = p
                    rrr = abs(entry["tp"] - entry["sl"]) / (abs(entry["tp"] - entry["sl"]) or 1)
                    result.append({
                        "ts":          int(entry["dt"].timestamp() * 1000),
                        "datum":       entry["dt"].isoformat(),
                        "strategie":   strat,
                        "side":        entry["side"],
                        "pnl":         pnl,
                        "equity":      1000.0,  # Wird später kumuliert
                        "hour":        entry["dt"].hour,
                        "weekday":     entry["dt"].weekday(),
                        "recentWR5":   50.0,
                        "recentWR15":  50.0,
                        "konsek":      0,
                        "rrr":         round(rrr, 2),
                        "ausgefuehrt": True,
                        "quelle":      "telegram",
                    })
                    del pending[strat]
                    break
            continue

        strategie = m_pnl.group(1).lower()
        pnl       = float(m_pnl.group(2))

        if strategie in pending:
            entry = pending.pop(strategie)
            sl, tp = entry["sl"], entry["tp"]
            rrr = abs(tp - sl) / (abs(tp - sl) or 1)
            result.append({
                "ts":          int(entry["dt"].timestamp() * 1000),
                "datum":       entry["dt"].isoformat(),
                "strategie":   strategie,
                "side":        entry["side"],
                "pnl":         pnl,
                "equity":      1000.0,
                "hour":        entry["dt"].hour,
                "weekday":     entry["dt"].weekday(),
                "recentWR5":   50.0,
                "recentWR15":  50.0,
                "konsek":      0,
                "rrr":         round(rrr, 2),
                "ausgefuehrt": True,
                "quelle":      "telegram",
            })

    # Equity kumulativ berechnen & Rolling-WRs nachrechnen
    by_strat = {}
    for t in result:
        by_strat.setdefault(t["strategie"], []).append(t)

    final = []
    for strat, trades in by_strat.items():
        trades.sort(key=lambda x: x["ts"])
        equity = 1000.0
        hist = []
        for t in trades:
            equity = round(equity + t["pnl"], 2)
            t["equity"] = equity
            hist.append(1 if t["pnl"] > 0 else 0)
            f5  = hist[-5:]
            f15 = hist[-15:]
            t["recentWR5"]  = round(sum(f5)/len(f5)*100, 1)   if f5  else 50.0
            t["recentWR15"] = round(sum(f15)/len(f15)*100, 1) if f15 else 50.0
            k = 0
            for v in reversed(hist[:-1]):
                if v == 0: k += 1
                else: break
            t["konsek"] = k
            final.append(t)

    return final


@app.post("/import-telegram")
async def import_telegram(
    file: UploadFile = File(...),
    retrain: bool = Form(True),
):
    content = (await file.read()).decode("utf-8", errors="replace")
    try:
        trades = parse_telegram_export(content)
    except Exception as e:
        raise HTTPException(400, f"Parse-Fehler: {e}")

    if not trades:
        raise HTTPException(400, "Keine Trades erkannt — prüfe ob die Nachrichten das richtige Format haben")

    features_file = DATA_DIR / "features.jsonl"
    existing_ts = set()
    if features_file.exists():
        with open(features_file) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    if t.get("quelle") == "telegram":
                        existing_ts.add(t.get("ts", 0))
                except:
                    pass

    neu = [t for t in trades if t["ts"] not in existing_ts]
    with open(features_file, "a") as f:
        for t in neu:
            f.write(json.dumps(t) + "\n")

    # Cache invalidieren
    for strat in set(t["strategie"] for t in neu):
        trades_cache.pop(strat, None)

    # Pro Strategie zusammenfassen
    by_strat = {}
    for t in neu:
        by_strat.setdefault(t["strategie"], 0)
        by_strat[t["strategie"]] += 1

    training = {}
    if retrain:
        for strat in by_strat:
            if len(lade_trades(strat)) >= MIN_TRADES:
                training[strat] = trainiere(strat)

    log.info(f"Telegram Import: {len(neu)} neue Trades — {by_strat}")
    return {
        "status":       "ok",
        "trades_neu":   len(neu),
        "trades_gesamt": len(trades),
        "pro_strategie": by_strat,
        "training":     training,
    }


@app.get("/import", response_class=HTMLResponse)
def import_page():
    """Einfache Upload-Seite für TradingView CSV"""
    opts = "\n".join(f'<option value="{s}">{s}</option>' for s in STRATEGIEN)
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>ML Import</title>
<style>
  body{{font-family:-apple-system,sans-serif;background:#0a0a0f;color:#e0e0f0;padding:40px;max-width:640px;margin:0 auto}}
  h1{{color:#f59e0b;margin-bottom:4px}}
  .tabs{{display:flex;gap:4px;margin:20px 0}}
  .tab{{padding:8px 18px;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600;background:#12121a;border:1px solid #1e1e2e;color:#555570}}
  .tab.active{{background:#f59e0b22;border-color:#f59e0b;color:#f59e0b}}
  .panel{{display:none}}.panel.active{{display:block}}
  p{{color:#555570;font-size:13px;margin-bottom:16px}}
  label{{display:block;font-size:12px;color:#555570;margin-bottom:6px;font-weight:600;margin-top:14px}}
  select,input{{width:100%;background:#12121a;border:1px solid #1e1e2e;color:#e0e0f0;padding:10px 14px;border-radius:8px;font-size:14px;box-sizing:border-box}}
  input[type=file]{{padding:8px}}input[type=checkbox]{{width:auto}}
  button{{margin-top:20px;width:100%;padding:12px;background:#f59e0b;color:#000;border:none;border-radius:8px;font-size:15px;font-weight:700;cursor:pointer}}
  .result{{margin-top:16px;padding:16px;background:#12121a;border-radius:8px;font-family:monospace;font-size:13px;display:none;white-space:pre-wrap}}
  .ok{{color:#22c55e}}.err{{color:#ef4444}}
  .step{{background:#12121a;border-left:3px solid #f59e0b;padding:10px 14px;margin-bottom:8px;border-radius:0 8px 8px 0;font-size:13px}}
  .step b{{color:#f59e0b}}
</style></head><body>
<h1>🤖 ML Import Center</h1>
<div class="tabs">
  <div class="tab active" onclick="show('tg',this)">📱 Telegram Export</div>
  <div class="tab" onclick="show('tv',this)">📊 TradingView CSV</div>
</div>

<!-- TELEGRAM -->
<div id="p-tg" class="panel active">
  <div class="step"><b>Schritt 1:</b> Telegram Desktop öffnen → den Bot-Chat öffnen</div>
  <div class="step"><b>Schritt 2:</b> Oben rechts ⋮ → "Chat exportieren" → Format: <b>JSON</b> → Exportieren</div>
  <div class="step"><b>Schritt 3:</b> Die <b>result.json</b> hier hochladen</div>
  <form id="f-tg">
    <label>JSON-Datei (Telegram Export)</label>
    <input type="file" name="file" accept=".json" required>
    <label style="display:flex;align-items:center;gap:8px;margin-top:14px">
      <input type="checkbox" name="retrain" checked> Direkt nach Import alle Modelle trainieren
    </label>
    <button type="submit">📤 Importieren & Trainieren</button>
  </form>
  <div class="result" id="r-tg"></div>
</div>

<!-- TRADINGVIEW -->
<div id="p-tv" class="panel">
  <div class="step"><b>Schritt 1:</b> TradingView → Strategie öffnen → Strategy Tester</div>
  <div class="step"><b>Schritt 2:</b> "Liste der Trades" → Export-Symbol (↓) → CSV</div>
  <div class="step"><b>Schritt 3:</b> CSV hier hochladen und Strategie auswählen</div>
  <form id="f-tv">
    <label>Strategie</label>
    <select name="strategie">{opts}</select>
    <label>Start-Equity (€)</label>
    <input type="number" name="start_equity" value="1000" step="100">
    <label>CSV-Datei</label>
    <input type="file" name="file" accept=".csv,.txt" required>
    <label style="display:flex;align-items:center;gap:8px;margin-top:14px">
      <input type=    <label style="display:flex;align-items:center;gap:8px;margin-top:14px">
      <input type="checkbox" name="retrain" checked> Direkt nach Import trainieren
    </label>
    <button type="submit">📤 Importieren & Trainieren</button>
  </form>
  <div class="result" id="r-tv"></div>
</div>

<script>
function show(id, el) {{
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('p-'+id).classList.add('active');
  el.classList.add('active');
}}

async function submitForm(formId, endpoint, resultId) {{
  const form = document.getElementById(formId);
  const fd = new FormData(form);
  fd.set('retrain', fd.get('retrain') ? 'true' : 'false');
  const btn = form.querySelector('button');
  btn.textContent = '\u23f3 Importiere...'; btn.disabled = true;
  const el = document.getElementById(resultId);
  el.style.display = 'block';
  try {{
    const r = await fetch(endpoint, {{method:'POST', body: fd}});
    const d = await r.json();
    if (r.ok) {{
      let html = `<span class="ok">\u2705 Erfolg!</span>\\n`;
      html += `Neue Trades: ${{d.trades_neu}} / ${{d.trades_gesamt}} gesamt\\n`;
      if (d.pro_strategie) html += `Pro Strategie: ${{JSON.stringify(d.pro_strategie)}}\\n`;
      if (d.training) {{
        html += `\\nTraining:\\n`;
        for (const [s,t] of Object.entries(d.training)) {{
          html += `  ${{s}}: ${{t.status}} \u2014 Accuracy ${{((t.accuracy_cv||0)*100).toFixed(1)}}% (${{t.n_trades}} Trades)\\n`;
        }}
      }}
      el.innerHTML = html;
    }} else {{
      el.innerHTML = `<span class="err">\u274c Fehler: ${{d.detail || JSON.stringify(d)}}</span>`;
    }}
  }} catch(err) {{
    el.innerHTML = `<span class="err">\u274c ${{err.message}}</span>`;
  }}
  btn.textContent = '\U0001f4e4 Importieren & Trainieren'; btn.disabled = false;
}}

document.getElementById('f-tg').onsubmit = e => {{ e.preventDefault(); submitForm('f-tg','/import-telegram','r-tg'); }};
document.getElementById('f-tv').onsubmit = e => {{ e.preventDefault(); submitForm('f-tv','/import-csv','r-tv'); }};
</script></body></html>"""

# \u2500\u2500 Startup \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
@app.on_event("startup")
def startup():
    lade_gespeicherte_modelle()
    # W\u00f6chentliches Auto-Retrain (jeden Montag 03:00)
    scheduler = BackgroundScheduler()
    scheduler.add_job(retrain_alle, "cron", day_of_week="mon", hour=3)
    scheduler.start()
    log.info("ML-Service gestartet")
