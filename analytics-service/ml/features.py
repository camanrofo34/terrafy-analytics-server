import numpy as np
import pandas as pd


def docs_to_dataframe(docs: list) -> pd.DataFrame:
    """Flatten MongoDB documents into a flat DataFrame."""
    rows = []
    for d in docs:
        rows.append({
            "timestamp":     d.get("timestamp"),
            "t_hours":       d.get("t_hours", 0),
            "temperature":   d.get("environment", {}).get("temperature_c"),
            "rh":            d.get("environment", {}).get("rh_percent"),
            "vpd":           d.get("environment", {}).get("vpd_kpa"),
            "vpd_status":    d.get("environment", {}).get("vpd_status"),
            "ph":            d.get("sensors", {}).get("ph"),
            "ec":            d.get("sensors", {}).get("ec_ms_cm"),
            "do":            d.get("sensors", {}).get("dissolved_o2"),
            "N":             d.get("concentrations", {}).get("N"),
            "P":             d.get("concentrations", {}).get("P"),
            "K":             d.get("concentrations", {}).get("K"),
            "root_length":   d.get("plant", {}).get("root_length_cm"),
            "growth_stage":  d.get("growth_stage"),
        })
    df = pd.DataFrame(rows).sort_values("t_hours").reset_index(drop=True)
    print(f"[DEBUG docs_to_dataframe] input docs={len(docs)}, output rows={len(df)}")
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        print(f"[DEBUG docs_to_dataframe] columns with NaN: {nan_cols.to_dict()}")
    return df


def engineer_features(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """
    Add rolling statistics and lag features.
    window = number of readings to look back (default 6 = 30s at 5s intervals).
    In production with hourly data, use window=6 for 6-hour rolling.
    """
    for col in ["temperature", "ph", "ec", "vpd", "N", "P", "K"]:
        df[f"{col}_mean{window}"]  = df[col].rolling(window, min_periods=1).mean()
        df[f"{col}_std{window}"]   = df[col].rolling(window, min_periods=1).std().fillna(0)
        df[f"{col}_lag1"]          = df[col].shift(1).fillna(df[col])
        df[f"{col}_delta"]         = df[col] - df[f"{col}_lag1"]

    engineered_cols = [c for c in df.columns if any(
        c.endswith(s) for s in ("_mean6", "_std6", "_lag1", "_delta")
    )]

    before = len(df)
    df = df.dropna(subset=engineered_cols)
    after = len(df)
    print(f"[DEBUG engineer_features] rows before dropna={before}, after={after}, dropped={before - after}")
    return df


# Feature columns used for each model
ALERT_FEATURES = [
    "temperature", "ph", "ec", "vpd", "do",
    "N", "P", "K",
    "temperature_mean6", "ph_mean6", "ec_mean6",
    "ph_std6", "ec_std6",
    "ph_delta", "ec_delta", "vpd_delta",
]

NUTRIENT_FEATURES = [
    "N", "P", "K", "t_hours", "root_length",
    "N_mean6", "P_mean6", "K_mean6",
    "N_delta", "P_delta", "K_delta",
    "growth_stage",
]

VPD_FEATURES = [
    "temperature", "rh", "vpd",
    "temperature_mean6", "temperature_std6",
    "temperature_delta",
]