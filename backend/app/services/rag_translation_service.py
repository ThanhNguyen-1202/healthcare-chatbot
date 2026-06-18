import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from app.core.config import settings

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


TRANSLATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "title_vi": {"type": "string"},
        "content_vi": {"type": "string"},
    },
    "required": ["title_vi", "content_vi"],
}


def _get_client():
    api_key = getattr(settings, "gemini_api_key", None) or os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)


def _get_model_name():
    return (
        getattr(settings, "gemini_translation_model", None)
        or getattr(settings, "gemini_model", None)
        or "gemini-3.1-flash-lite"
    )


def _build_prompt(title: str, content: str) -> str:
    return f"""
Bạn là trợ lý y khoa cho chatbot sàng lọc ban đầu.

Nhiệm vụ:
- Đọc tài liệu tiếng Anh bên dưới
- Dịch và tóm tắt sang tiếng Việt thật dễ hiểu
- Dịch phần tiêu đề (title) chính xác và có thể có các từ đồng nghĩa phổ biến
- Không chẩn đoán chắc chắn
- Không thêm thông tin ngoài tài liệu
- Viết ngắn gọn, tối đa 6 câu cho phần nội dung

Tiêu đề:
{title}

Nội dung:
{content}
""".strip()


def _normalize_translation_result(data: Dict[str, Any], original_title: str, original_content: str) -> Dict[str, str]:
    title_vi = data.get("title_vi")
    content_vi = data.get("content_vi")

    if not isinstance(title_vi, str) or not title_vi.strip():
        title_vi = original_title

    if not isinstance(content_vi, str) or not content_vi.strip():
        content_vi = original_content

    return {
        "title_vi": title_vi.strip(),
        "content_vi": content_vi.strip(),
    }


@lru_cache(maxsize=256)
def translate_rag_doc_to_vietnamese(title: str, content: str) -> dict:
    if not title and not content:
        return {
            "title_vi": "",
            "content_vi": "",
        }

    client = _get_client()
    if client is None or types is None:
        return {
            "title_vi": title,
            "content_vi": content,
        }

    try:
        prompt = _build_prompt(title, content[:5000])

        response = client.models.generate_content(
            model=_get_model_name(),
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=TRANSLATION_SCHEMA,
                temperature=0.2,
            ),
        )

        raw_text = (response.text or "").strip()
        data = json.loads(raw_text)

        if not isinstance(data, dict):
            return {
                "title_vi": title,
                "content_vi": content,
            }

        return _normalize_translation_result(data, title, content)

    except TimeoutError as exc:
        logger.error("Gemini RAG translation timed out: %s", exc)
        return {
            "title_vi": title,
            "content_vi": content,
        }