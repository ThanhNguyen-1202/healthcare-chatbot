from typing import Optional

from app.core.vietnam_time import now_vietnam
from app.db.mongo import get_database


class SessionRepository:
    def __init__(self):
        self.collection_name = "chat_sessions"

    def _get_collection(self):
        db = get_database()
        return db[self.collection_name]

    def _build_empty_session(
        self,
        session_id: str,
        device_id: Optional[str] = None,
    ) -> dict:
        now = now_vietnam()

        return {
            "session_id": session_id,
            "device_id": device_id,  # [CHANGED] Map session với Device ID khi tạo session mới.
            "status": "active",  # [CHANGED] active | completed để biết phiên đã kết thúc chưa.
            "created_at": now,
        }

    async def create_session_if_not_exists(
        self,
        session_id: str,
        device_id: Optional[str] = None,
    ) -> None:
        collection = self._get_collection()
        now = now_vietnam()

        await collection.update_one(
            {"session_id": session_id},
            {
                "$setOnInsert": {
                    **self._build_empty_session(session_id, device_id),
                    "messages": [],
                    "intake_snapshot": None,
                },
                "$set": {
                    "updated_at": now,  # [CHANGED] Không set device_id ở đây để tránh conflict MongoDB.
                },
            },
            upsert=True,
        )


    async def attach_device_if_missing(
        self,
        session_id: str,
        device_id: Optional[str],
    ) -> None:
        """Attach a device_id to legacy sessions that were created before device binding."""
        if not device_id:
            return

        collection = self._get_collection()
        now = now_vietnam()

        await collection.update_one(
            {
                "session_id": session_id,
                "$or": [
                    {"device_id": {"$exists": False}},
                    {"device_id": None},
                    {"device_id": ""},
                ],
            },
            {
                "$set": {
                    "device_id": device_id,
                    "updated_at": now,
                }
            },
        )

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        collection = self._get_collection()
        now = now_vietnam()

        await collection.update_one(
            {"session_id": session_id},
            {
                "$setOnInsert": {
                    "session_id": session_id,
                    "device_id": None,
                    "status": "active",
                    "intake_snapshot": None,
                    "created_at": now,
                },
                "$push": {
                    "messages": {
                        "role": role,
                        "content": content,
                        "timestamp": now,
                    }
                },
                "$set": {
                    "updated_at": now,
                },
            },
            upsert=True,
        )

    async def get_session(self, session_id: str) -> Optional[dict]:
        collection = self._get_collection()
        return await collection.find_one({"session_id": session_id})

    async def mark_session_completed(self, session_id: str) -> None:
        """Mark a session as completed after final screening output."""
        collection = self._get_collection()
        now = now_vietnam()

        await collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "status": "completed",  # Khóa session sau khi đã trả kết quả cuối.
                    "completed_at": now,
                    "updated_at": now,
                }
            },
        )

    async def update_intake_snapshot(
        self,
        session_id: str,
        intake_data: dict,
    ) -> None:
        collection = self._get_collection()
        now = now_vietnam()

        await collection.update_one(
            {"session_id": session_id},
            {
                "$setOnInsert": {
                    **self._build_empty_session(session_id),
                    "messages": [],
                },
                "$set": {
                    "intake_snapshot": intake_data or {},
                    "updated_at": now,
                },
            },
            upsert=True,
        )

    async def clear_intake_snapshot(self, session_id: str) -> None:
        collection = self._get_collection()
        now = now_vietnam()

        await collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "intake_snapshot": None,
                    "updated_at": now,
                }
            },
        )