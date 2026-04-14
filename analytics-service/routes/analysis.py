import os
from fastapi import APIRouter, HTTPException, Query
from pymongo import MongoClient
from datetime import datetime, timezone, timedelta
from typing import Optional

from registry import VARIABLES, SYSTEMS, ALERT_RULES, GROUPING_UNITS, get_nested

router = APIRouter()
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
col = client[os.getenv("MONGO_DB", "terrafy")]["readings"]


def _utc(dt: datetime) -> datetime:
    """Ensure datetime is UTC-aware."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Registry info
# ---------------------------------------------------------------------------

@router.get("/variables")
def list_variables():
    """Return all registered agronomic variables."""
    return list(VARIABLES.values())


@router.get("/systems")
def list_systems():
    """Return all registered agronomic systems."""
    return list(SYSTEMS.values())


# ---------------------------------------------------------------------------
# Latest snapshot (kept for backward compat)
# ---------------------------------------------------------------------------

@router.get("/latest")
def get_latest(system_id: Optional[str] = None):
    query = {"system_id": system_id} if system_id else {}
    doc = col.find_one(query, sort=[("timestamp", -1)])
    if doc:
        doc.pop("_id")
    return doc or {}


# ---------------------------------------------------------------------------
# Historic data — single variable for a system
# ---------------------------------------------------------------------------

@router.get("/history")
def get_history(
    system_id: str,
    variable_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    grouping: str = Query("hours", enum=["minutes", "hours", "days", "weeks"]),
):
    """
    Return time-grouped statistics for one agronomic variable on one system.

    - system_id:    e.g. 1, 2, 3
    - variable_id:  e.g. 1 (Temperatura), 4 (pH), 5 (EC) …
    - start_date / end_date: ISO-8601 (defaults to last 24 h)
    - grouping:      minutes | hours | days | weeks
    """
    if variable_id not in VARIABLES:
        raise HTTPException(
            400,
            detail=f"Unknown variable '{variable_id}'. Valid: {list(VARIABLES.keys())}",
        )
    if system_id not in SYSTEMS:
        raise HTTPException(
            400,
            detail=f"Unknown system '{system_id}'. Valid: {list(SYSTEMS.keys())}",
        )

    var = VARIABLES[variable_id]
    now = datetime.now(timezone.utc)
    start = _utc(start_date) if start_date else now - timedelta(hours=24)
    end   = _utc(end_date)   if end_date   else now
    unit  = GROUPING_UNITS[grouping]

    pipeline = [
        {"$match": {"system_id": system_id, "timestamp": {"$gte": start, "$lte": end}}},
        {"$addFields": {
            "bucket": {"$dateTrunc": {"date": "$timestamp", "unit": unit}}
        }},
        {"$group": {
            "_id":   "$bucket",
            "avg":   {"$avg": f"${var['field']}"},
            "min":   {"$min": f"${var['field']}"},
            "max":   {"$max": f"${var['field']}"},
            "count": {"$sum": 1},
        }},
        {"$sort": {"_id": 1}},
        {"$project": {
            "_id": 0,
            "timestamp": "$_id",
            "avg":   {"$round": ["$avg",   4]},
            "min":   {"$round": ["$min",   4]},
            "max":   {"$round": ["$max",   4]},
            "count": 1,
        }},
    ]

    data = list(col.aggregate(pipeline))
    return {
        "system_id":   system_id,
        "system_name": SYSTEMS[system_id]["name"],
        "variable":    var,
        "grouping":    grouping,
        "start_date":  start,
        "end_date":    end,
        "data":        data,
    }


# ---------------------------------------------------------------------------
# Historic data — all variables for a system
# ---------------------------------------------------------------------------

@router.get("/history/{system_id}")
def get_history_by_system(
    system_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    grouping: str = Query("hours", enum=["minutes", "hours", "days", "weeks"]),
):
    """
    Return time-grouped statistics for every agronomic variable of a system.
    """
    if system_id not in SYSTEMS:
        raise HTTPException(400, detail=f"Unknown system '{system_id}'")

    now   = datetime.now(timezone.utc)
    start = _utc(start_date) if start_date else now - timedelta(hours=24)
    end   = _utc(end_date)   if end_date   else now
    unit  = GROUPING_UNITS[grouping]

    match = {"system_id": system_id, "timestamp": {"$gte": start, "$lte": end}}

    variables_data = {}
    for var_name, var in VARIABLES.items():
        pipeline = [
            {"$match": match},
            {"$addFields": {
                "bucket": {"$dateTrunc": {"date": "$timestamp", "unit": unit}}
            }},
            {"$group": {
                "_id":   "$bucket",
                "avg":   {"$avg": f"${var['field']}"},
                "min":   {"$min": f"${var['field']}"},
                "max":   {"$max": f"${var['field']}"},
                "count": {"$sum": 1},
            }},
            {"$sort": {"_id": 1}},
            {"$project": {
                "_id": 0,
                "timestamp": "$_id",
                "avg":   {"$round": ["$avg",   4]},
                "min":   {"$round": ["$min",   4]},
                "max":   {"$round": ["$max",   4]},
                "count": 1,
            }},
        ]
        variables_data[var_name] = {
            "variable": var,
            "data":     list(col.aggregate(pipeline)),
        }

    return {
        "system_id":   system_id,
        "system_name": SYSTEMS[system_id]["name"],
        "grouping":    grouping,
        "start_date":  start,
        "end_date":    end,
        "variables":   variables_data,
    }


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@router.get("/alerts")
def get_alerts(
    system_id: str,
    variable_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
):
    """
    Return threshold violations for a system, optionally filtered to one variable.

    Alertable variable IDs: 3 (VPD), 4 (pH), 5 (EC), 7 (Nitrógeno).
    """
    if system_id not in SYSTEMS:
        raise HTTPException(400, detail=f"Unknown system '{system_id}'")
    if variable_id and variable_id not in ALERT_RULES:
        raise HTTPException(
            400,
            detail=f"Variable '{variable_id}' has no alert rules. Alertable IDs: {list(ALERT_RULES.keys())}",
        )

    now   = datetime.now(timezone.utc)
    start = _utc(start_date) if start_date else now - timedelta(hours=24)
    end   = _utc(end_date)   if end_date   else now

    docs = list(col.find(
        {"system_id": system_id, "timestamp": {"$gte": start, "$lte": end}},
        {"_id": 0},
    ))

    rules = (
        {variable_id: ALERT_RULES[variable_id]}
        if variable_id
        else ALERT_RULES
    )

    alerts = []
    for doc in docs:
        for vid, rule in rules.items():
            val = get_nested(doc, rule["field"])
            if val is None:
                continue
            direction = None
            if rule["min"] is not None and val < rule["min"]:
                direction = "below_minimum"
            elif rule["max"] is not None and val > rule["max"]:
                direction = "above_maximum"
            if direction:
                alerts.append({
                    "type":          f"variable_{vid}_out_of_range",
                    "variable_id":   vid,
                    "variable_name": VARIABLES[vid]["name"],
                    "value":         round(val, 4),
                    "unit":          rule["unit"],
                    "direction":     direction,
                    "system_id":     system_id,
                    "timestamp":     doc["timestamp"],
                })

    return {
        "system_id":   system_id,
        "system_name": SYSTEMS[system_id]["name"],
        "start_date":  start,
        "end_date":    end,
        "alert_count": len(alerts),
        "alerts":      alerts[-50:],
    }
