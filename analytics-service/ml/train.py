"""
Run this script to train all models:
    python -m ml.train

Requires at least a few hundred readings in MongoDB.
The more simulation data you have, the better the accuracy.
"""

import os, pickle, pathlib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

from ml.features import docs_to_dataframe, engineer_features
from ml.features import ALERT_FEATURES, NUTRIENT_FEATURES, VPD_FEATURES

load_dotenv()

MODELS_DIR = pathlib.Path("ml/models")
MODELS_DIR.mkdir(exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
col    = client[os.getenv("MONGO_DB", "terrafy")]["readings"]

print("Loading data from MongoDB...")
docs = list(col.find({}, {"_id": 0}))
print(f"  {len(docs)} documents loaded.")

if len(docs) < 100:
    print("Not enough data. Run the simulation engine for longer and try again.")
    exit(1)

df = docs_to_dataframe(docs)
df = engineer_features(df)
print(f"  {len(df)} rows after feature engineering.")


# ── 1. Alert classifier ───────────────────────────────────────────────────────
# Label: which parameter is out of range (multiclass)
def make_alert_label(row):
    if row["vpd_status"] == "too_high":  return "vpd_high"
    if row["vpd_status"] == "too_low":   return "vpd_low"
    if row["ph"] < 5.5:                  return "ph_low"
    if row["ph"] > 6.5:                  return "ph_high"
    if row["ec"] < 1.2:                  return "ec_low"
    if row["ec"] > 2.5:                  return "ec_high"
    if row["N"] < 1.0:                   return "N_critical"
    return "normal"

df["alert_label"] = df.apply(make_alert_label, axis=1)
print(f"\nAlert label distribution:\n{df['alert_label'].value_counts()}")

X_alert = df[ALERT_FEATURES].values
y_alert = df["alert_label"].values

le_alert = LabelEncoder()
y_enc    = le_alert.fit_transform(y_alert)

X_tr, X_te, y_tr, y_te = train_test_split(X_alert, y_enc, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)

cv_scores = cross_val_score(clf, X_alert, y_enc, cv=5, scoring="accuracy")
print(f"\nAlert classifier — CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(classification_report(y_te, clf.predict(X_te), target_names=le_alert.classes_))

pickle.dump(clf,      open(MODELS_DIR / "alert_clf.pkl",    "wb"))
pickle.dump(le_alert, open(MODELS_DIR / "alert_labels.pkl", "wb"))


# ── 2. Nutrient depletion regressor ──────────────────────────────────────────
# Target: hours until N drops below critical threshold (1.0 mmol/L)
# We compute this from the simulation trajectory
df_sorted = df.sort_values("t_hours").copy()

# For each row, find how many hours until N hits threshold
N_CRITICAL = 1.0
def hours_until_critical(idx, df_sorted):
    current_t = df_sorted.loc[idx, "t_hours"]
    future = df_sorted[df_sorted["t_hours"] > current_t]
    critical = future[future["N"] <= N_CRITICAL]
    if critical.empty:
        return 999.0   # won't deplete in this simulation window
    return round(critical.iloc[0]["t_hours"] - current_t, 2)

df_sorted["hours_to_N_critical"] = [
    hours_until_critical(i, df_sorted) for i in df_sorted.index
]

# Encode growth_stage for this model
le_stage = LabelEncoder()
df_sorted["stage_enc"] = le_stage.fit_transform(df_sorted["growth_stage"])
feat_nut = [f for f in NUTRIENT_FEATURES if f != "growth_stage"] + ["stage_enc"]

X_nut = df_sorted[feat_nut].values
y_nut = df_sorted["hours_to_N_critical"].values

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_nut, y_nut, test_size=0.2, random_state=42)

reg = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
reg.fit(X_tr2, y_tr2)

mae = mean_absolute_error(y_te2, reg.predict(X_te2))
print(f"\nNutrient depletion regressor — Test MAE: {mae:.2f} hours")

pickle.dump(reg,      open(MODELS_DIR / "nutrient_reg.pkl",   "wb"))
pickle.dump(le_stage, open(MODELS_DIR / "stage_encoder.pkl",  "wb"))


# ── 3. VPD next-reading predictor ────────────────────────────────────────────
# Predicts VPD N steps ahead (regression)
STEPS_AHEAD = 6

df_vpd = df.copy()
df_vpd["vpd_future"] = df_vpd["vpd"].shift(-STEPS_AHEAD)
df_vpd = df_vpd.dropna(subset=["vpd_future"])

X_vpd = df_vpd[VPD_FEATURES].values
y_vpd = df_vpd["vpd_future"].values

X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X_vpd, y_vpd, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_tr3s  = scaler.fit_transform(X_tr3)
X_te3s  = scaler.transform(X_te3)

vpd_reg = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=42)
vpd_reg.fit(X_tr3s, y_tr3)

mae3 = mean_absolute_error(y_te3, vpd_reg.predict(X_te3s))
print(f"\nVPD predictor — Test MAE: {mae3:.4f} kPa  (predicting {STEPS_AHEAD} steps ahead)")

pickle.dump(vpd_reg, open(MODELS_DIR / "vpd_reg.pkl",    "wb"))
pickle.dump(scaler,  open(MODELS_DIR / "vpd_scaler.pkl", "wb"))


# ── 4. Intent classifier (chatbot) ───────────────────────────────────────────
# Training data: (example question, intent label)
INTENT_EXAMPLES = [
    # status
    ("what is the current status",           "status"),
    ("how is the system right now",          "status"),
    ("show me current readings",             "status"),
    ("what are the sensor values",           "status"),
    ("system overview",                      "status"),
    ("how is everything",                    "status"),
    ("give me a summary",                    "status"),

    # alerts
    ("are there any alerts",                 "alerts"),
    ("what problems do you see",             "alerts"),
    ("any warnings",                         "alerts"),
    ("is something wrong",                   "alerts"),
    ("what is out of range",                 "alerts"),
    ("show me alerts",                       "alerts"),
    ("any issues right now",                 "alerts"),
    ("check for problems",                   "alerts"),

    # predictions
    ("what will happen next",                "predict"),
    ("predict the next few hours",           "predict"),
    ("when will nutrients run out",          "predict"),
    ("how long until nitrogen is critical",  "predict"),
    ("forecast vpd",                         "predict"),
    ("what will the temperature be",         "predict"),
    ("predict nutrient depletion",           "predict"),
    ("future readings",                      "predict"),

    # trend
    ("show me the trend",                    "trend"),
    ("how has ph changed",                   "trend"),
    ("is temperature rising",                "trend"),
    ("what happened in the last hour",       "trend"),
    ("historical data",                      "trend"),
    ("how did ec evolve",                    "trend"),
    ("last 24 hours summary",                "trend"),
    ("show history",                         "trend"),

    # recommendation
    ("what should i do",                     "recommend"),
    ("give me a recommendation",             "recommend"),
    ("how can i fix this",                   "recommend"),
    ("what action should i take",            "recommend"),
    ("advice for the system",                "recommend"),
    ("how to improve",                       "recommend"),
    ("what do you suggest",                  "recommend"),
    ("any recommendations",                  "recommend"),

    # growth
    ("what growth stage is the plant",       "growth"),
    ("how big are the roots",                "growth"),
    ("plant status",                         "growth"),
    ("how is the lettuce growing",           "growth"),
    ("root length",                          "growth"),
    ("biomass",                              "growth"),
]

texts   = [e[0] for e in INTENT_EXAMPLES]
intents = [e[1] for e in INTENT_EXAMPLES]

vectorizer    = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
X_intent      = vectorizer.fit_transform(texts)
le_intent     = LabelEncoder()
y_intent      = le_intent.fit_transform(intents)

intent_clf = LogisticRegression(max_iter=1000, C=5.0)
intent_clf.fit(X_intent, y_intent)

cv_intent = cross_val_score(intent_clf, X_intent, y_intent, cv=3, scoring="accuracy")
print(f"\nIntent classifier — CV accuracy: {cv_intent.mean():.3f} ± {cv_intent.std():.3f}")

pickle.dump(intent_clf,  open(MODELS_DIR / "intent_clf.pkl",    "wb"))
pickle.dump(vectorizer,  open(MODELS_DIR / "intent_tfidf.pkl",  "wb"))
pickle.dump(le_intent,   open(MODELS_DIR / "intent_labels.pkl", "wb"))

print("\nAll models saved to ml/models/. Training complete.")