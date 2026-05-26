import logging
import pandas as pd
from pymongo import MongoClient, ASCENDING, ReplaceOne
from pymongo.errors import ConnectionFailure, BulkWriteError
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
        logger.info(f"MongoDB connected: DB={settings.MONGODB_DB}")
    except ConnectionFailure as e:
        logger.warning(f"MongoDB connection warning: {e}")


def insert_sales_data(df: pd.DataFrame):
    col = get_db()["sales"]
    operations = []
    for row in df.to_dict(orient="records"):
        try:
            doc = {
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
            }
            operations.append(
                ReplaceOne(
                    {"product_id": doc["product_id"], "date": doc["date"], "store_id": doc["store_id"]},
                    doc,
                    upsert=True,
                )
            )
        except Exception as e:
            logger.warning(f"Skipping row: {e}")

    if operations:
        try:
            result = col.bulk_write(operations, ordered=False)
            logger.info(
                f"Upserted {result.upserted_count} new, modified {result.modified_count} existing records."
            )
        except BulkWriteError as bwe:
            logger.warning(f"Bulk write partial error: {bwe.details}")
