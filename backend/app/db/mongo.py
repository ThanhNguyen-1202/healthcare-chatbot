import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import settings

logger = logging.getLogger(__name__)


class MongoDB:

    client: Optional[AsyncIOMotorClient] = None


mongodb = MongoDB()


async def connect_to_mongo() -> None:
    if not settings.mongo_uri:
        raise RuntimeError("MONGO_URI is required. InMemoryDatabase fallback is disabled.")

    mongodb.client = AsyncIOMotorClient(
        settings.mongo_uri,
        serverSelectionTimeoutMS=5000,
        uuidRepresentation="standard",
    )
    await mongodb.client.admin.command("ping")
    logger.info("Connected to MongoDB database: %s", settings.mongo_db)


async def close_mongo_connection() -> None:
    if mongodb.client is not None:
        mongodb.client.close()
        mongodb.client = None
        logger.info("MongoDB connection closed")


def get_database() -> AsyncIOMotorDatabase:
    if mongodb.client is None:
        raise RuntimeError("MongoDB client is not connected. Startup connection failed.")
    return mongodb.client[settings.mongo_db]
