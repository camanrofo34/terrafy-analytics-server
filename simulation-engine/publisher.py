import os
from datetime import datetime, timezone
from pymongo import MongoClient

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client[os.getenv("MONGO_DB", "terrafy")]
readings_col = db["readings"]


def publish_reading(reading: dict):
    reading["timestamp"] = datetime.now(timezone.utc)
    readings_col.insert_one(reading)