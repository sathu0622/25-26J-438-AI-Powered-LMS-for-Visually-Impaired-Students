import os
from datetime import datetime
from typing import Any, Dict, List

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config import (
    MONGODB_DB_NAME,
    MONGODB_FAVORITES_COLLECTION,
    MONGODB_SERVER_SELECTION_TIMEOUT_MS,
    MONGODB_TLS_CA_FILE,
    MONGODB_TLS_INSECURE,
    MONGODB_URI,
)


def _mongo_client_kwargs() -> Dict[str, Any]:
    """TLS options for MongoDB Atlas (Windows/OpenSSL often needs an explicit CA bundle)."""
    kwargs: Dict[str, Any] = {
        "serverSelectionTimeoutMS": MONGODB_SERVER_SELECTION_TIMEOUT_MS,
    }
    uri_lower = MONGODB_URI.lower()
    if MONGODB_TLS_CA_FILE and os.path.isfile(MONGODB_TLS_CA_FILE):
        kwargs["tlsCAFile"] = MONGODB_TLS_CA_FILE
    elif "mongodb+srv://" in uri_lower or ".mongodb.net" in uri_lower:
        try:
            import certifi

            kwargs["tlsCAFile"] = certifi.where()
        except ImportError:
            pass
    if MONGODB_TLS_INSECURE:
        kwargs["tlsAllowInvalidCertificates"] = True
    return kwargs


class FavoriteArticlesStore:
    """MongoDB-backed store for globally shared favorite articles."""

    def __init__(self):
        self.client = MongoClient(MONGODB_URI, **_mongo_client_kwargs())
        self.collection = self.client[MONGODB_DB_NAME][MONGODB_FAVORITES_COLLECTION]
        self.collection.create_index(
            [("document_id", 1), ("article_id", 1)],
            unique=True,
            name="unique_document_article",
        )
        self.collection.create_index("created_at")

    def add_favorite(self, article: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        full_content = article.get("full_content") or ""
        summary = article.get("summary") or ""
        payload = {
            "document_id": article["document_id"],
            "article_id": article["article_id"],
            "heading": article.get("heading", "No heading"),
            "subheading": article.get("subheading", ""),
            "body_preview": article.get("body_preview", ""),
            "full_content": full_content,
            "summary": summary,
            "resource_type": article.get("resource_type", ""),
            "created_at": now,
            "updated_at": now,
        }

        self.collection.update_one(
            {"document_id": payload["document_id"], "article_id": payload["article_id"]},
            {
                "$set": {
                    "heading": payload["heading"],
                    "subheading": payload["subheading"],
                    "body_preview": payload["body_preview"],
                    "full_content": payload["full_content"],
                    "summary": payload["summary"],
                    "resource_type": payload["resource_type"],
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "document_id": payload["document_id"],
                    "article_id": payload["article_id"],
                    "created_at": now,
                },
            },
            upsert=True,
        )
        return payload

    def list_favorites(self) -> List[Dict[str, Any]]:
        docs = list(self.collection.find({}, {"_id": 0}).sort("created_at", -1))
        return docs

    def remove_favorite(self, document_id: str, article_id: str) -> int:
        result = self.collection.delete_one(
            {"document_id": document_id, "article_id": article_id}
        )
        return result.deleted_count


def get_favorite_store() -> FavoriteArticlesStore:
    try:
        return FavoriteArticlesStore()
    except PyMongoError as exc:
        raise RuntimeError(f"MongoDB connection failed: {exc}") from exc
