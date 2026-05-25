import logging
import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure
from backend.config import settings

logger = logging.getLogger(__name__)

_client = None
_db = None


def get_client():
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
    return _client


def get_db():
    global _db
    if _db is None:
        _db = get_client()[settings.MONGODB_DB]
    return _db


def init_db():
    try:
        db = get_db()
        db["sales"].create_index([("product_id", ASCENDING), ("date", ASCENDING)])
        logger.info(f"MongoDB connected: {settings.MONGODB_URI} | DB: {settings.MONGODB_DB}")
    except ConnectionFailure as e:
        logger.warning(f"MongoDB connection warning: {e}")


def insert_sales_data(df: pd.DataFrame):
    col = get_db()["sales"]
    records = []
    for row in df.to_dict(orient="records"):
        try:
            records.append({
                "date": str(pd.to_datetime(row["date"]).date()),
                "product_id": str(row["product_id"]),
                "product_name": str(row.get("product_name", "")),
                "category": str(row.get("category", "")),
                "price": float(row.get("price", 0)),
                "discount": float(row.get("discount", 0)),
                "units_sold": int(row.get("units_sold", 0)),
                "revenue": float(row.get("revenue", 0)),
                "store_id": str(row.get("store_id", "")),
                "region": str(row.get("region", "")),
            })
        except Exception as e:
            logger.warning(f"Skipping row: {e}")

    if records:
        col.delete_many({})
        col.insert_many(records)
        logger.info(f"Inserted {len(records)} records into MongoDB.")
