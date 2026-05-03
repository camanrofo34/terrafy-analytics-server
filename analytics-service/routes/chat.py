import os
import pickle
import pathlib
from datetime import datetime, timezone, timedelta

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient

from ml.features import docs_to_dataframe, engineer_features
from ml.features import ALERT_FEATURES, NUTRIENT_FEATURES, VPD_FEATURES

router = APIRouter()
MODELS_DIR = pathlib.Path("ml/models")

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
col = client[os.getenv("MONGO_DB", "terrafy")]["readings"]


def load_model(name):
    p = MODELS_DIR / name
    if not p.exists():
        return None
    return pickle.load(open(p, "rb"))


alert_clf = load_model("alert_clf.pkl")
alert_labels = load_model("alert_labels.pkl")
nutrient_reg = load_model("nutrient_reg.pkl")
stage_enc = load_model("stage_encoder.pkl")
vpd_reg = load_model("vpd_reg.pkl")
vpd_scaler = load_model("vpd_scaler.pkl")
intent_clf = load_model("intent_clf.pkl")
intent_tfidf = load_model("intent_tfidf.pkl")
intent_labels = load_model("intent_labels.pkl")

MODELS_READY = all([
    alert_clf,
    nutrient_reg,
    vpd_reg,
    intent_clf,
    intent_tfidf,
    intent_labels,
])

INTENT_ES = {
    "status": "estado",
    "alerts": "alertas",
    "predict": "prediccion",
    "trend": "tendencia",
    "recommend": "recomendacion",
    "growth": "crecimiento",
}

STAGE_ES = {
    "seedling": "plantula",
    "vegetative": "vegetativa",
    "mature": "madura",
    "harvest_ready": "lista_para_cosecha",
    "unknown": "desconocida",
}

VPD_STATUS_ES = {
    "low": "bajo",
    "optimal": "optimo",
    "normal": "normal",
    "high": "alto",
}

ALERT_LABEL_ES = {
    "normal": "normal",
    "vpd_high": "vpd_alto",
    "vpd_low": "vpd_bajo",
    "ph_high": "ph_alto",
    "ph_low": "ph_bajo",
    "ec_high": "ec_alta",
    "ec_low": "ec_baja",
    "N_critical": "N_critico",
}


class ChatRequest(BaseModel):
    message: str
    system_id: str = "1"


def detect_intent(message: str) -> tuple[str, float]:
    if not MODELS_READY:
        return "status", 0.0
    vec = intent_tfidf.transform([message.lower()])
    proba = intent_clf.predict_proba(vec)[0]
    idx = np.argmax(proba)
    return intent_labels.inverse_transform([idx])[0], round(float(proba[idx]), 3)


def get_recent_df(hours=6, simulation_id="1"):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    print(f"[DEBUG get_recent_df] querying simulation_id={simulation_id!r}, since={since.isoformat()}")
    docs = list(
        col.find({"system_id": simulation_id, "timestamp": {"$gte": since}}).sort("timestamp", 1)
    )
    print(f"[DEBUG get_recent_df] MongoDB returned {len(docs)} document(s)")
    if not docs:
        print("[DEBUG get_recent_df] No docs found, returning None. Check simulation_id and that the simulation is running.")
        return None, None
    df = docs_to_dataframe(docs)
    df = engineer_features(df)
    print(f"[DEBUG get_recent_df] final df shape={df.shape}, latest timestamp={docs[-1].get('timestamp')}")
    return df, docs[-1]


def handle_status(df, latest):
    if latest is None:
        return "Aun no hay datos disponibles. Verifica que el motor de simulacion este en ejecucion."

    env = latest.get("environment", {})
    sens = latest.get("sensors", {})
    conc = latest.get("concentrations", {})
    plant = latest.get("plant", {})

    stage_raw = latest.get("growth_stage", "unknown")
    stage_es = STAGE_ES.get(stage_raw, stage_raw)
    vpd_status_raw = env.get("vpd_status", "?")
    vpd_status_es = VPD_STATUS_ES.get(vpd_status_raw, vpd_status_raw)

    return (
        f"**Estado actual del sistema** ({latest['timestamp'].strftime('%H:%M:%S UTC')})\n\n"
        f"Etapa de crecimiento: {stage_es} | "
        f"Longitud de raiz: {plant.get('root_length_cm', '?')} cm\n\n"
        f"**Ambiente**\n"
        f"Temperatura: {env.get('temperature_c', '?')} C | "
        f"Humedad: {env.get('rh_percent', '?')} % | "
        f"VPD: {env.get('vpd_kpa', '?')} kPa ({vpd_status_es})\n\n"
        f"**Sensores**\n"
        f"pH: {sens.get('ph', '?')} | "
        f"EC: {sens.get('ec_ms_cm', '?')} mS/cm | "
        f"Oxigeno disuelto: {sens.get('dissolved_o2', '?')} mg/L\n\n"
        f"**Concentraciones de nutrientes**\n"
        f"N: {conc.get('N', '?')} mmol/L | "
        f"P: {conc.get('P', '?')} mmol/L | "
        f"K: {conc.get('K', '?')} mmol/L"
    )


def handle_alerts(df, latest):
    if df is None or not MODELS_READY:
        return "No hay datos para analizar alertas."

    row = df.iloc[-1]
    X = np.array([[row.get(f, 0) for f in ALERT_FEATURES]])
    pred = alert_clf.predict(X)[0]
    label = alert_labels.inverse_transform([pred])[0]
    proba = alert_clf.predict_proba(X)[0].max()

    alert_messages = {
        "normal": "Todos los parametros estan dentro del rango optimo.",
        "vpd_high": "VPD demasiado alto (>2.0 kPa). Las plantas tienen estres termico. Considera subir humedad o bajar temperatura.",
        "vpd_low": "VPD demasiado bajo (<0.5 kPa). La transpiracion esta suprimida. Reduce la humedad.",
        "ph_high": "pH demasiado alto (>6.5). Se reduce la disponibilidad de fosforo y hierro. Agrega solucion para bajar pH.",
        "ph_low": "pH demasiado bajo (<5.5). Riesgo de bloqueo de nutrientes. Agrega solucion para subir pH.",
        "ec_high": "EC demasiado alta (>2.5 mS/cm). Riesgo de estres osmotico. Diluye la solucion nutritiva.",
        "ec_low": "EC demasiado baja (<1.2 mS/cm). Las plantas estan subalimentadas. Agrega concentrado de nutrientes.",
        "N_critical": "Nitrogeno criticamente bajo (<1.0 mmol/L). Reponer de inmediato.",
    }

    label_es = ALERT_LABEL_ES.get(label, label)
    msg = alert_messages.get(label, f"Alerta detectada: {label_es}")
    return f"{msg}\n\n*Confianza del modelo: {proba:.0%}*"


def handle_predict(df, latest):
    if df is None or not MODELS_READY:
        return "Todavia no hay suficientes datos para generar predicciones."

    messages = []

    row = df.iloc[-1]
    X_vpd = np.array([[row.get(f, 0) for f in VPD_FEATURES]])
    X_vpds = vpd_scaler.transform(X_vpd)
    vpd_pred = vpd_reg.predict(X_vpds)[0]
    current_vpd = row.get("vpd", 0)
    direction = "suba a" if vpd_pred > current_vpd else "baje a"

    messages.append(
        f"**Pronostico de VPD (proximos ~30s con el tick actual):** "
        f"Se espera que {direction} {vpd_pred:.3f} kPa "
        f"(actualmente {current_vpd:.3f} kPa)."
    )

    feat_nut = [f for f in NUTRIENT_FEATURES if f != "growth_stage"] + ["stage_enc"]
    stage_val = latest.get("growth_stage", "vegetative") if latest else "vegetative"
    try:
        stage_encoded = stage_enc.transform([stage_val])[0]
    except Exception:
        stage_encoded = 1

    row_nut = {f: row.get(f, 0) for f in feat_nut if f != "stage_enc"}
    row_nut["stage_enc"] = stage_encoded
    X_nut = np.array([[row_nut.get(f, 0) for f in feat_nut]])
    hours_left = nutrient_reg.predict(X_nut)[0]

    if hours_left > 500:
        messages.append(
            "**Agotamiento de nitrogeno:** Los niveles del tanque son estables y no se espera agotamiento critico en esta ventana de simulacion."
        )
    else:
        messages.append(
            f"**Pronostico de agotamiento de nitrogeno:** Con las tasas de absorcion actuales, el nitrogeno llegara a nivel critico en aproximadamente **{hours_left:.1f} horas**."
        )

    return "\n\n".join(messages)


def handle_trend(df, latest):
    print(f"[DEBUG handle_trend] df={'None' if df is None else f'shape={df.shape}'}")
    if df is None or len(df) < 3:
        print(
            f"[DEBUG handle_trend] BLOCKED - df is None={df is None}, len={0 if df is None else len(df)} (need >= 3)"
        )
        return "Aun no hay suficientes datos historicos para analizar tendencias."

    results = []
    for col_name, label, unit in [
        ("ph", "pH", ""),
        ("temperature", "Temperatura", "C"),
        ("ec", "EC", "mS/cm"),
        ("vpd", "VPD", "kPa"),
    ]:
        if col_name not in df.columns:
            continue

        series = df[col_name].dropna()
        delta = series.iloc[-1] - series.iloc[0]
        trend = "subiendo" if delta > 0.05 else ("bajando" if delta < -0.05 else "estable")
        results.append(
            f"**{label}:** {trend} "
            f"(de {series.iloc[0]:.3f} a {series.iloc[-1]:.3f} {unit} en {len(df)} lecturas)"
        )

    return "**Analisis de tendencia (ultimas 6 horas)**\n\n" + "\n".join(results)


def handle_recommend(df, latest):
    alert_text = handle_alerts(df, latest)
    rec_text = handle_predict(df, latest)
    return (
        "**Recomendaciones segun condiciones actuales**\n\n"
        f"{alert_text}\n\n"
        f"**Contexto de pronostico**\n{rec_text}"
    )


def handle_growth(df, latest):
    if latest is None:
        return "No hay datos disponibles."

    plant = latest.get("plant", {})
    stage = latest.get("growth_stage", "unknown")
    stage_es = STAGE_ES.get(stage, stage)
    t = latest.get("t_hours", 0)
    prl = plant.get("root_length_cm", 0)

    stage_tips = {
        "seedling": "Las raices aun se estan estableciendo. Mantener VPD bajo (0.8-1.2 kPa) para reducir estres por trasplante.",
        "vegetative": "Fase de crecimiento activo. Mantener EC en 1.6-2.0 mS/cm y asegurar nitrogeno suficiente.",
        "mature": "Cerca del tamano final. Reducir un poco nitrogeno y vigilar tipburn (deficiencia de calcio).",
        "harvest_ready": "La planta esta lista para cosecha. Se puede subir ligeramente EC para mejorar concentracion de sabor.",
    }

    tip = stage_tips.get(stage, "Monitorear cuidadosamente todos los parametros.")
    return (
        "**Estado de crecimiento de la planta**\n\n"
        f"Etapa actual: **{stage_es}** (t = {t:.1f} h desde trasplante)\n"
        f"Longitud de raiz: **{prl} cm**\n\n"
        f"{tip}"
    )


HANDLERS = {
    "status": handle_status,
    "alerts": handle_alerts,
    "predict": handle_predict,
    "trend": handle_trend,
    "recommend": handle_recommend,
    "growth": handle_growth,
}


@router.post("/chat")
def chat(req: ChatRequest):
    if not MODELS_READY:
        raise HTTPException(503, detail="ML models not trained yet. Run: python -m ml.train")

    intent, confidence = detect_intent(req.message)
    print(
        f"[DEBUG chat] message={req.message!r}, system_id={req.system_id!r}, intent={intent}, confidence={confidence}"
    )

    df, latest = get_recent_df(hours=6, simulation_id=req.system_id)
    handler = HANDLERS.get(intent, handle_status)
    response_text = handler(df, latest)

    return {
        "system_id": req.system_id,
        "intent": INTENT_ES.get(intent, intent),
        "confidence": confidence,
        "response": response_text,
    }
