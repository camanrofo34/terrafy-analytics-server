import os
from fastapi import APIRouter, HTTPException
from pymongo import MongoClient
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from typing import Optional

from registry import VARIABLES, SYSTEMS, ALERT_RULES, get_nested

router = APIRouter()
client_db = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
col = client_db[os.getenv("MONGO_DB", "terrafy")]["readings"]

AGRONOMIC_SYSTEM_PROMPT = (
    "You are an agronomic AI assistant for the Terrafy hydroponic monitoring platform. "
    "You have access to real-time and historic sensor data from multiple hydroponic growing systems. "
    "Answer questions concisely and professionally, focusing on plant health, nutrient management, "
    "and environmental conditions. When you detect issues, provide actionable recommendations "
    "grounded in the data provided. Keep responses brief unless the user asks for detail."
)


class ChatRequest(BaseModel):
    message: str
    system_id: str


# ---------------------------------------------------------------------------
# Rule-based recommendation (original endpoint, enhanced with system_id)
# ---------------------------------------------------------------------------

@router.get("/recommendation")
def get_recommendation(system_id: str = "1"):
    if system_id not in SYSTEMS:
        raise HTTPException(400, detail=f"Unknown system '{system_id}'")

    doc = col.find_one({"system_id": system_id}, sort=[("timestamp", -1)])
    if not doc:
        return {"recommendation": "no data available"}

    recs = []
    vpd_val = get_nested(doc, "environment.vpd_kpa")
    ph_val  = get_nested(doc, "sensors.ph")
    ec_val  = get_nested(doc, "sensors.ec_ms_cm")
    n_conc  = get_nested(doc, "concentrations.N")

    if vpd_val is not None:
        if vpd_val > 2.0:
            recs.append("Increase humidity — high VPD causing stomatal stress.")
        elif vpd_val < 0.5:
            recs.append("Reduce humidity — low VPD suppressing transpiration.")

    if ph_val is not None:
        if ph_val < 5.5:
            recs.append("Add pH-up solution — nutrient lockout risk below 5.5.")
        elif ph_val > 6.5:
            recs.append("Add pH-down solution — reduced P and Fe availability above 6.5.")

    if ec_val is not None:
        if ec_val < 1.2:
            recs.append("Increase nutrient concentration — EC too low for current growth stage.")
        elif ec_val > 2.5:
            recs.append("Dilute solution — EC too high, osmotic stress risk.")

    if n_conc is not None and n_conc < 1.0:
        recs.append("Replenish nitrogen — tank concentration critically low.")

    return {
        "system_id":       system_id,
        "system_name":     SYSTEMS[system_id]["name"],
        "timestamp":       doc["timestamp"],
        "growth_stage":    doc.get("growth_stage"),
        "recommendations": recs if recs else ["All parameters within optimal range."],
    }


# ---------------------------------------------------------------------------
# AI chatbot — powered by Claude
# ---------------------------------------------------------------------------

@router.post("/chat")
def chat(req: ChatRequest):
    """
    Ask the AI assistant anything about the historic data of a system.
    Requires ANTHROPIC_API_KEY to be set in the environment.
    """
    if req.system_id not in SYSTEMS:
        raise HTTPException(400, detail=f"Unknown system '{req.system_id}'")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            503,
            detail="ANTHROPIC_API_KEY is not configured. Set it in .env.local to enable the chatbot.",
        )

    try:
        import anthropic
    except ImportError:
        raise HTTPException(503, detail="anthropic package not installed.")

    # --- Build data context from the last 24 h ---
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    docs  = list(
        col.find(
            {"system_id": req.system_id, "timestamp": {"$gte": since}},
            {"_id": 0},
        )
        .sort("timestamp", -1)
        .limit(200)
    )

    system_name = SYSTEMS[req.system_id]["name"]

    if not docs:
        context = f"System: {system_name} ({req.system_id})\nNo data available in the last 24 hours."
    else:
        latest = docs[0]
        lines = [
            f"System: {system_name} ({req.system_id})",
            f"Latest reading: {latest['timestamp']}",
            f"Growth stage: {latest.get('growth_stage', 'unknown')}",
            f"Temperature: {get_nested(latest, 'environment.temperature_c')} °C",
            f"Humidity: {get_nested(latest, 'environment.rh_percent')} %",
            f"VPD: {get_nested(latest, 'environment.vpd_kpa')} kPa "
            f"({get_nested(latest, 'environment.vpd_status')})",
            f"pH: {get_nested(latest, 'sensors.ph')}",
            f"EC: {get_nested(latest, 'sensors.ec_ms_cm')} mS/cm",
            f"Dissolved O₂: {get_nested(latest, 'sensors.dissolved_o2')} mg/L",
            f"Nitrogen: {get_nested(latest, 'concentrations.N')} mmol/L",
            f"Phosphorus: {get_nested(latest, 'concentrations.P')} mmol/L",
            f"Potassium: {get_nested(latest, 'concentrations.K')} mmol/L",
            f"Root length: {get_nested(latest, 'plant.root_length_cm')} cm",
            "",
            f"Data window: last 24 h ({len(docs)} readings)",
        ]

        # 24-hour ranges for key variables
        for var_key, field in [
            ("temperature", "environment.temperature_c"),
            ("ph",          "sensors.ph"),
            ("ec",          "sensors.ec_ms_cm"),
        ]:
            values = [get_nested(d, field) for d in docs if get_nested(d, field) is not None]
            if values:
                lines.append(
                    f"24h {var_key}: min={min(values):.3f}  max={max(values):.3f}"
                    f"  avg={sum(values)/len(values):.3f}"
                )

        context = "\n".join(lines)

    user_message = f"System data context:\n{context}\n\nUser question: {req.message}"

    claude = anthropic.Anthropic(api_key=api_key)
    response = claude.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=1024,
        system=AGRONOMIC_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return {
        "system_id":      req.system_id,
        "system_name":    system_name,
        "response":       response.content[0].text,
        "input_tokens":   response.usage.input_tokens,
        "output_tokens":  response.usage.output_tokens,
    }
