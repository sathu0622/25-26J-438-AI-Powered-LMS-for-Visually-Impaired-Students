from datetime import datetime
from typing import Dict, Any, List

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config import (
    MONGODB_URI,
    MONGODB_DB_NAME,
    MONGODB_FAVORITES_COLLECTION,
)


class FavoriteArticlesStore:
    """MongoDB-backed store for globally shared favorite articles."""

    def __init__(self):
        self.client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        self.collection = self.client[MONGODB_DB_NAME][MONGODB_FAVORITES_COLLECTION]
        self.collection.create_index(
            [("document_id", 1), ("article_id", 1)],
            unique=True,
            name="unique_document_article",
        )
        self.collection.create_index("created_at")

    def add_favorite(self, article: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        payload = {
            "document_id": article["document_id"],
            "article_id": article["article_id"],
            "heading": article.get("heading", "No heading"),
            "subheading": article.get("subheading", ""),
            "body_preview": article.get("body_preview", ""),
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
