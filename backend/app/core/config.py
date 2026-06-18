

from pathlib import Path
from typing import List

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"


class Settings(BaseSettings):


    app_name: str = "healthcare-chatbot"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    mongo_uri: str = Field(default="", alias="MONGO_URI")
    mongo_db: str = Field(default="Chatbox_Healthcare", alias="MONGO_DB")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.1-flash-lite", alias="GEMINI_MODEL")
    gemini_extractor_model: str = Field(default="gemini-3.1-flash-lite", alias="GEMINI_EXTRACTOR_MODEL")
    gemini_translation_model: str = Field(default="gemini-3.1-flash-lite", alias="GEMINI_TRANSLATION_MODEL")

    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:5500", "http://127.0.0.1:5500"],
        alias="CORS_ORIGINS",
    )
    chat_rate_limit: str = Field(default="20/minute", alias="CHAT_RATE_LIMIT")
    predict_rate_limit: str = Field(default="30/minute", alias="PREDICT_RATE_LIMIT")

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> List[str]:
        """Accept CORS origins as CSV string or JSON-like list."""
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("CORS_ORIGINS must be a comma-separated string or list")

    @model_validator(mode="after")
    def validate_required_secrets(self) -> "Settings":
        """Crash early when critical production dependencies are not configured."""
        missing = []
        if not self.mongo_uri.strip():
            missing.append("MONGO_URI")
        if not self.gemini_api_key.strip() or self.gemini_api_key == "your_gemini_api_key_here":
            missing.append("GEMINI_API_KEY")
        if not self.cors_origins:
            missing.append("CORS_ORIGINS")
        if missing:
            raise ValueError(
                "Missing required environment variables: " + ", ".join(missing)
            )
        return self


settings = Settings()
