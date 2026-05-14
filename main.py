import os, json, pickle, logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

MIN_TRADES   = int(os.getenv("MIN_TRADES", "30"))   # Mindest-Trades für Training
PREDICT_CONF = float(os.getenv("PREDICT_CONF", "0.58"))  # Mindest-Konfidenz für Trade

STRATEGIEN = ["mittel", "aggressiv", "smart", "konservativ", "optimiert", "test", "adaptive", "steady"]

# ── State ─────────────────────────────────────────────
models: dict = {}        # { strategie: Pipeline }
model_meta: dict = {}    # { strategie: { trainiert_am, accuracy, n_trades } }
trades_cache: dict = {}  # { strategie: [trades] }

# ── Daten laden ───────────────────────────────────────
def lade_trades(strategie: str) -> list:
    """Lädt Trades aus features.jsonl (bevorzugt) oder trades.json"""
    features_file = DATA_DIR / "features.jsonl"
    trades_file   = DATA_DIR / "trades.json"

    trades = []

    # 1. Feature-Log (reichhaltigste Daten)
    if features_file.exists():
        with open(features_file) as f:
            for line in f:
                try:
                    t = json.loads(line.strip())
                    if t.get("strategie") == strategie and t.get("ausgefuehrt"):
                        trades.append(t)
                except:
                    pass

    # 2. Fallback: trades.json
    if not trades and trades_file.exists():
        with open(trades_file) as f:
            all_trades = json.load(f)
        raw = all_trades.get(strategie, [])
        for t in raw:
            trades.append({
                "pnl":      t.get("pnl", 0),
                "side":     t.get("side", "BUY"),
                "equity":   t.get("equity", 1000),
                "hour":     datetime.fromisoformat(t["datum"]).hour if "datum" in t else 12,
                "weekday":  datetime.fromisoformat(t["datum"]).weekday() if "datum" in t else 0,
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

        row = {
            "hour":        t.get("hour", 12),
            "weekday":     t.get("weekday", 0),
            "side_buy":    1 if t.get("side", "BUY") == "BUY" else 0,
            "wr5":         round(wr5, 3),
            "wr15":        round(wr15, 3),
            "konsek":      min(konsek, 10),
            "equity":      t.get("equity", 1000),
            "rrr":         t.get("rrr") or 2.0,
            "recentWR5":   t.get("recentWR5") or round(wr5 * 100, 1),
            "recentWR15":  t.get("recentWR15") or round(wr15 * 100, 1),
            "label":       1 if pnl > 0 else 0,
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

    # Pipeline: Skalierung + Random Forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
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

# ── API Endpoints ─────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelle": list(models.keys()),
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

    empfehlung = "trade" if win_prob >= PREDICT_CONF else "skip"
    grund = (
        f"ML: {win_prob:.0%} Gewinn-Wahrscheinlichkeit (Schwelle: {PREDICT_CONF:.0%})"
    )

    log.info(f"[{req.strategie}] Predict: {req.side} → {empfehlung} ({win_prob:.0%})")
    return {
        "empfehlung": empfehlung,
        "konfidenz":  konfidenz,
        "win_prob":   round(win_prob * 100, 1),
        "schwelle":   round(PREDICT_CONF * 100, 1),
        "grund":      grund,
        "trainiert":  True,
    }

@app.get("/feature-importance/{strategie}")
def feature_importance(strategie: str):
    if strategie not in models:
        raise HTTPException(404, "Kein Modell für diese Strategie")
    rf = models[strategie].named_steps["model"]
    names = ["hour","weekday","side_buy","wr5","wr15","konsek","equity","rrr","recentWR5","recentWR15"]
    importance = sorted(zip(names, rf.feature_importances_), key=lambda x: -x[1])
    return {"strategie": strategie, "importance": [{"feature": n, "wert": round(float(v),4)} for n,v in importance]}

# ── Startup ───────────────────────────────────────────
@app.on_event("startup")
def startup():
    lade_gespeicherte_modelle()
    # Wöchentliches Auto-Retrain (jeden Montag 03:00)
    scheduler = BackgroundScheduler()
    scheduler.add_job(retrain_alle, "cron", day_of_week="mon", hour=3)
    scheduler.start()
    log.info("ML-Service gestartet")
