from typing import Any, Dict, List

from app.core.vietnam_time import now_vietnam
from app.db.mongo import get_database


class PredictionRepository:
    def __init__(self):
        self.collection_name = "predictions"

    def _get_collection(self):
        db = get_database()
        return db[self.collection_name]

    async def save_prediction(
        self,
        session_id: str,
        collected_data: dict,
        prediction_result: dict,
    ):
        collection = self._get_collection()

        document = {
            "session_id": session_id,
            "collected_data": collected_data or {},
            "prediction_result": prediction_result or {},
            "created_at": now_vietnam(),
        }

        await collection.insert_one(document)

    async def get_predictions_by_session(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        collection = self._get_collection()
        cursor = collection.find({"session_id": session_id}).sort("created_at", -1)
        return await cursor.to_list(length=limit)

    async def get_latest_prediction_by_session(self, session_id: str):
        collection = self._get_collection()
        return await collection.find_one(
            {"session_id": session_id},
            sort=[("created_at", -1)]
        )