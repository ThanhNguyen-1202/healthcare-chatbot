import logging
import re
import uuid
from typing import Dict, List, Optional, Set

from app.schemas.chat import ChatRequest, ChatResponse
from app.db.repositories.session_repo import SessionRepository
from app.db.repositories.prediction_repo import PredictionRepository

from app.services.intake_service import build_intake_from_messages, empty_intake
from app.services.llm_extractor_service import extract_intake_with_llm
from app.services.prediction_service import predict_from_symptoms
from app.services.ml_disease_prediction_service import predict_top_diseases
from app.services.hybrid_disease_prediction_service import predict_hybrid_diseases
from app.services.ddxplus_evidence_catalog import normalize_evidence_codes
from app.services.emergency_triage_service import check_emergency_triage
from app.services.department_service import infer_department_from_diseases, infer_department_from_symptoms
from app.services.triage_service import get_action_guidance

from app.rag.retriever import retrieve_relevant_documents
from app.rag.explainer import build_rag_explanation

logger = logging.getLogger(__name__)


FIELD_LABELS = {
    "age": "tuổi",
    "gender": "giới tính",
    "main_symptom": "triệu chứng chính",
    "duration": "thời gian khởi phát/kéo dài",
    "temperature": "nhiệt độ",
    "pain_score": "điểm đau 0-10",
    "symptom_location": "vị trí triệu chứng",
    "secondary_symptoms": "triệu chứng kèm theo",
    "red_flags": "dấu hiệu nguy hiểm",
    "comorbidities": "bệnh nền",
    "medications": "thuốc đang dùng",
    "is_pregnant": "tình trạng mang thai",
}

FIELD_QUESTIONS = {
    "main_symptom": "Anh/chị vui lòng nhập triệu chứng chính hiện tại. Ví dụ: sốt, ho, đau bụng, đau ngực, phát ban hoặc khó thở.",
    "duration": "Triệu chứng này đã xuất hiện hoặc kéo dài bao lâu rồi?",
    "temperature": "Anh/chị có đo nhiệt độ không? Nếu có, hiện tại khoảng bao nhiêu độ?",
    "pain_score": "Nếu có đau, anh/chị chấm điểm đau từ 0 đến 10 là bao nhiêu?",
    "symptom_location": "Triệu chứng nằm ở vị trí nào rõ nhất? Ví dụ: ngực trái, bụng dưới, vùng trán hoặc sau hốc mắt.",
    "red_flags": "Anh/chị có dấu hiệu nguy hiểm nào không: khó thở nặng, đau ngực dữ dội, lơ mơ, ngất, co giật, chảy máu nhiều hoặc nôn liên tục? Nếu không có, trả lời 'không'.",
    "comorbidities": "Anh/chị có bệnh nền gì không? Ví dụ: tiểu đường, tăng huyết áp, hen, bệnh tim hoặc suy thận. Nếu không có, trả lời 'không'.",
    "medications": "Anh/chị đang dùng hoặc mới uống thuốc gì không? Ví dụ: paracetamol, aspirin, kháng sinh hoặc thuốc huyết áp. Nếu không có, trả lời 'không'.",
    "age": "Anh/chị vui lòng cho biết tuổi của người đang có triệu chứng. Ví dụ: 25 tuổi.",
    "gender": "Anh/chị vui lòng cho biết giới tính của người đang có triệu chứng: nam hoặc nữ.",
    "is_pregnant": "Nếu anh/chị là nữ, hiện có đang mang thai không? Nếu không liên quan, có thể trả lời 'không'.",
    "secondary_symptoms": "Anh/chị còn triệu chứng nào khác không? Ví dụ: ho, khó thở, buồn nôn, đau đầu hoặc phát ban. Nếu không có, trả lời 'không'.",
}

REQUIRED_FIELDS = ["age", "gender", "main_symptom"]

# Không bắt buộc hỏi thêm các trường khác. Hệ thống chỉ chặn dự đoán khi thiếu
# tuổi, giới tính hoặc triệu chứng chính theo yêu cầu hiện tại.
OPTIONAL_FIELDS = []

QUESTION_ORDER = ["age", "gender", "main_symptom"]


def is_missing_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def strip_internal_fields(data: Optional[dict]) -> dict:
    clean = {}
    for key, value in (data or {}).items():
        if not str(key).startswith("_"):
            clean[key] = value
    return clean


def ensure_intake_shape(data: Optional[dict]) -> dict:
    shaped = empty_intake()
    shaped.update(data or {})
    shaped.setdefault("secondary_symptoms", [])
    shaped.setdefault("red_flags", [])
    shaped.setdefault("comorbidities", [])
    shaped.setdefault("medications", [])
    shaped.setdefault("ddxplus_evidences", [])
    return shaped


def get_optional_answered_fields(collected_data: dict) -> Set[str]:
    values = collected_data.get("_optional_answered_fields", []) or []
    return {item for item in values if isinstance(item, str)}


def mark_optional_answered(collected_data: dict, field_name: str) -> dict:
    answered = list(get_optional_answered_fields(collected_data))
    if field_name not in answered:
        answered.append(field_name)
    collected_data["_optional_answered_fields"] = answered
    return collected_data


def should_skip_optional_field(field_name: str, collected_data: dict) -> bool:
    return field_name in get_optional_answered_fields(collected_data)


def is_negative_reply(text: str) -> bool:
    if not isinstance(text, str):
        return False

    normalized = text.lower().strip()
    negative_phrases = {
        "không", "không ạ", "không có", "không còn", "không thêm", "không có thêm",
        "không có gì thêm", "không có triệu chứng nào khác", "hết rồi", "không biết",
        "không rõ", "chưa đo", "chưa biết", "không dùng", "không uống", "không mang thai",
    }
    return normalized in negative_phrases


def merge_unique_list(base: list, extra: list, exclude: Optional[str] = None) -> list:
    results = list(base or [])
    for item in extra or []:
        if not isinstance(item, str):
            continue
        clean = item.lower().strip()
        if clean and clean != exclude and clean not in results:
            results.append(clean)
    return results


COMMON_SYMPTOM_KEYWORDS = [
    "sốt", "sốt cao", "ho", "ho khan", "ho có đờm", "đau họng", "sổ mũi", "nghẹt mũi",
    "khó thở", "thở khò khè", "đau ngực", "tức ngực", "đánh trống ngực", "hồi hộp",
    "tim đập nhanh", "vã mồ hôi", "chóng mặt", "mệt mỏi", "sụt cân", "tăng cân",
    "đau đầu", "đau nửa đầu", "nhìn mờ", "nói khó", "yếu liệt", "tê tay", "tê chân",
    "co giật", "ngất", "lơ mơ", "lú lẫn", "hay quên", "giảm trí nhớ", "khó tập trung",
    "mất định hướng", "đặt đồ vật sai vị trí", "không nhớ", "đau bụng", "đau dạ dày",
    "buồn nôn", "nôn", "nôn ói", "tiêu chảy", "táo bón", "đầy bụng", "chướng bụng",
    "đi ngoài ra máu", "nôn ra máu", "đau quặn bụng", "tiểu buốt", "tiểu rắt",
    "tiểu nhiều", "tiểu ra máu", "nước tiểu đục", "dịch trắng đục", "đau lưng",
    "đau khớp", "sưng khớp", "cứng khớp", "đau cơ", "phát ban", "nổi mẩn", "ngứa",
    "da khô", "nổi mề đay", "mụn nước", "sưng ở cổ", "khó nuốt", "khàn tiếng",
    "tuyến giáp phình to", "khối u ở cổ", "run tay", "rậm lông", "ra máu bất thường",
    "chảy máu nhiều", "đau vùng chậu", "khí hư", "đau khi quan hệ"
]


WEAK_MAIN_SYMPTOMS = {
    "đau", "mệt", "mệt mỏi", "khó chịu", "không khỏe", "triệu chứng", "bệnh", "sốt", "ho"
}

CANONICAL_SYMPTOM_RULES = [
    (r"đặt đồ.*sai|không nhớ.*để|quên.*đồ", "hay quên, giảm trí nhớ"),
    (r"vòng eo lớn|béo bụng|tăng cân.*đói|đói bụng.*tăng cân", "tăng cân, vòng eo lớn, thường xuyên đói bụng"),
    (r"sưng.*cổ|khối u.*cổ|tuyến giáp.*phình|bướu cổ", "sưng ở cổ, khó nuốt, khàn tiếng"),
    (r"hồi hộp|tim đập nhanh|run tay|sụt cân", "hồi hộp, tim đập nhanh, run tay, sụt cân"),
    (r"tiểu buốt|tiểu rát|dịch trắng đục|dịch.*lỗ sáo", "tiểu buốt, tiểu rát, dịch trắng đục"),
    (r"ngứa|da khô|nổi mẩn|phát ban|mề đay", "ngứa, da khô, nổi mẩn"),
    (r"đau quặn bụng|tiêu chảy ra máu|sút cân", "đau quặn bụng, tiêu chảy ra máu, sút cân"),
    (r"ra máu.*sau sinh|chảy máu.*sau sinh|ra máu nhiều", "ra máu nhiều bất thường"),
    (r"rậm lông|kinh nguyệt không đều|mụn trứng cá", "rậm lông, kinh nguyệt không đều, mụn trứng cá"),
    (r"khó ngủ|mất ngủ|lo âu|căng thẳng", "mất ngủ, lo âu, căng thẳng"),
    (r"đau vùng chậu|khí hư|đau khi quan hệ", "đau vùng chậu, khí hư, đau khi quan hệ"),
]


def is_weak_main_symptom(value: Optional[str]) -> bool:
    if not isinstance(value, str):
        return True
    text = value.lower().strip()
    if not text:
        return True
    if text in WEAK_MAIN_SYMPTOMS:
        return True
    return len(text) < 4


def normalize_symptom_phrase(value: str) -> str:
    text = value.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(tôi|em|mình|cháu|anh|chị)\s+", "", text)
    text = re.sub(r"^(hiện đang|đang|có|bị|thấy|hay bị|thường xuyên)\s+", "", text)
    text = re.sub(r"\b(tôi|em|mình|cháu)\s+có\s+thể\s+đang\s+bị\s+bệnh\s+gì.*$", "", text)
    text = re.sub(r"\b(có\s+thể\s+là\s+bệnh\s+gì|là\s+bệnh\s+gì|bị\s+bệnh\s+gì).*$", "", text)
    text = re.sub(r"\b(trong|khoảng|kéo dài|đã kéo dài|từ)\s+\d+\s*(giờ|ngày|tuần|tháng|năm).*$", "", text)
    text = text.strip(" ,.;:-?!")
    return text


def split_symptom_items(value: str) -> List[str]:
    if not isinstance(value, str):
        return []
    text = normalize_symptom_phrase(value)
    if not text:
        return []
    parts = []
    for item in re.split(r"[,;/]|\s+và\s+|\s+kèm\s+|\s+cùng\s+", text):
        clean = normalize_symptom_phrase(item)
        if clean and clean not in parts:
            parts.append(clean)
    return parts


def join_unique_symptoms(*values) -> str:
    parts = []
    for value in values:
        if isinstance(value, str):
            candidates = split_symptom_items(value)
        elif isinstance(value, (list, tuple, set)):
            candidates = []
            for item in value:
                if isinstance(item, str):
                    candidates.extend(split_symptom_items(item))
        else:
            candidates = []
        for item in candidates:
            if item and item not in parts:
                parts.append(item)
    return ", ".join(parts[:16])


def semantic_main_symptom_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    semantic_hits = []
    for pattern, canonical in CANONICAL_SYMPTOM_RULES:
        if re.search(pattern, normalized):
            semantic_hits.append(canonical)
    if semantic_hits:
        return join_unique_symptoms(semantic_hits)
    return None


def build_llm_context_text(messages: List[dict], latest_user_message: str) -> str:
    user_messages = []
    for item in messages[-8:]:
        if item.get("role") == "user" and item.get("content"):
            user_messages.append(str(item.get("content")))
    if latest_user_message and latest_user_message not in user_messages:
        user_messages.append(latest_user_message)
    conversation_text = "\n".join(user_messages[-6:])
    return (
        "Phân tích nội dung người dùng bên dưới như dữ liệu triệu chứng y khoa. "
        "Hãy quy đổi mô tả tự nhiên thành triệu chứng chính ngắn gọn bằng tiếng Việt. "
        "Nếu có nhiều triệu chứng, hãy gom vào main_symptom, không bỏ sót triệu chứng. "
        "Ví dụ: đặt đồ sai vị trí -> hay quên, giảm trí nhớ; "
        "sưng ở cổ khó nuốt khàn tiếng -> sưng cổ, khó nuốt, khàn tiếng; "
        "tiểu buốt tiểu rát dịch trắng đục -> tiểu buốt, tiểu rát, dịch trắng đục.\n\n"
        f"Nội dung người dùng:\n{conversation_text}"
    )


def is_llm_intake_useful(data: Optional[dict]) -> bool:
    """Return True only when the intake is useful for DDXPlus prediction.

    A main symptom alone is not enough for the structured classifier; at least
    one valid DDXPlus evidence code must survive normalization.
    """
    shaped = ensure_intake_shape(data)
    evidence_codes = normalize_evidence_codes(
        shaped.get("ddxplus_evidences", [])
    )
    if not evidence_codes:
        return False
    if not is_weak_main_symptom(shaped.get("main_symptom")):
        return True
    if shaped.get("secondary_symptoms") or shaped.get("red_flags"):
        return True
    return False


def extract_intake_with_llm_boosted(latest_user_message: str, messages: List[dict]) -> dict:
    first_result = extract_intake_with_llm(latest_user_message)
    if is_llm_intake_useful(first_result):
        return first_result

    boosted_text = build_llm_context_text(messages, latest_user_message)
    second_result = extract_intake_with_llm(boosted_text)
    return merge_intake_data(first_result, second_result)


def clean_fallback_symptom_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    value = normalize_symptom_phrase(text)
    if len(value) < 3:
        return None

    return value[:220]


def extract_symptom_keywords_from_text(text: str) -> List[str]:
    if not isinstance(text, str):
        return []

    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)

    found = []
    for keyword in COMMON_SYMPTOM_KEYWORDS:
        if keyword in normalized and keyword not in found:
            found.append(keyword)

    return found


def fallback_main_symptom_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)

    semantic_value = semantic_main_symptom_from_text(normalized)
    if semantic_value:
        return semantic_value

    patterns = [
        r"các triệu chứng như\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"triệu chứng như\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"triệu chứng\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"tôi hiện đang có\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"em hiện đang có\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"tôi đang có\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"em đang có\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"tôi đang bị\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"em đang bị\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"tôi bị\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"em bị\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"tôi thấy\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"em thấy\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"tôi hay bị\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"em hay bị\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"người bệnh có\s+(.+?)(?:\.|,?\s*có thể|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            value = clean_fallback_symptom_text(match.group(1))
            if value:
                return join_unique_symptoms(value)

    keywords = extract_symptom_keywords_from_text(normalized)
    if keywords:
        return join_unique_symptoms(keywords[:16])

    health_words = [
        "đau", "sốt", "ho", "khó thở", "ngứa", "sưng", "buồn nôn", "tiêu chảy",
        "tiểu", "chảy máu", "khàn tiếng", "khó nuốt", "run tay", "sụt cân",
        "mệt", "phát ban", "hồi hộp", "tim đập nhanh", "hay quên", "không nhớ",
        "dịch", "máu", "đói", "tăng cân", "giảm cân", "tê", "yếu", "lú lẫn"
    ]

    if any(word in normalized for word in health_words) and len(normalized) >= 10:
        return clean_fallback_symptom_text(normalized)

    return None


def combine_main_and_secondary_symptoms(collected_data: dict) -> dict:
    collected_data = ensure_intake_shape(collected_data)

    parts = []

    main_symptom = collected_data.get("main_symptom")
    if isinstance(main_symptom, str) and main_symptom.strip():
        for item in re.split(r"[,;/]| và ", main_symptom):
            clean = item.strip()
            if clean and clean not in parts:
                parts.append(clean)

    for symptom in collected_data.get("secondary_symptoms", []) or []:
        if isinstance(symptom, str):
            clean = symptom.strip()
            if clean and clean not in parts:
                parts.append(clean)

    if parts:
        collected_data["main_symptom"] = ", ".join(parts)

    collected_data["secondary_symptoms"] = []
    return collected_data


def enhance_main_symptom_with_fallback(collected_data: dict, latest_user_message: str) -> dict:
    collected_data = ensure_intake_shape(collected_data)
    fallback_value = fallback_main_symptom_from_text(latest_user_message)

    if not fallback_value and isinstance(latest_user_message, str):
        fallback_value = clean_fallback_symptom_text(latest_user_message)

    if fallback_value:
        if is_weak_main_symptom(collected_data.get("main_symptom")):
            collected_data["main_symptom"] = fallback_value
        else:
            collected_data["main_symptom"] = join_unique_symptoms(collected_data.get("main_symptom"), fallback_value)

    collected_data = combine_main_and_secondary_symptoms(collected_data)
    return collected_data


def merge_intake_data(rule_data: Optional[dict], llm_data: Optional[dict]) -> dict:
    merged = ensure_intake_shape(rule_data)
    llm_data = ensure_intake_shape(llm_data)

    scalar_fields = [
        "age", "gender", "duration", "temperature", "pain_score",
        "symptom_location", "is_pregnant",
    ]

    for key in scalar_fields:
        if is_missing_value(merged.get(key)) and not is_missing_value(llm_data.get(key)):
            merged[key] = llm_data.get(key)

    current_main = merged.get("main_symptom")
    candidate_main = llm_data.get("main_symptom")

    if is_weak_main_symptom(current_main) and not is_weak_main_symptom(candidate_main):
        merged["main_symptom"] = candidate_main
    elif not is_missing_value(candidate_main) and candidate_main != current_main:
        merged["secondary_symptoms"] = merge_unique_list(
            merged.get("secondary_symptoms", []),
            [candidate_main],
            exclude=current_main,
        )

    main_symptom = merged.get("main_symptom")
    merged["secondary_symptoms"] = merge_unique_list(
        merged.get("secondary_symptoms", []),
        llm_data.get("secondary_symptoms", []),
        exclude=main_symptom,
    )
    merged["red_flags"] = merge_unique_list(merged.get("red_flags", []), llm_data.get("red_flags", []))
    merged["comorbidities"] = merge_unique_list(merged.get("comorbidities", []), llm_data.get("comorbidities", []))
    merged["medications"] = merge_unique_list(merged.get("medications", []), llm_data.get("medications", []))
    evidence_codes = normalize_evidence_codes(
        list(merged.get("ddxplus_evidences", []) or [])
        + list(llm_data.get("ddxplus_evidences", []) or [])
    )
    merged["ddxplus_evidences"] = evidence_codes

    return combine_main_and_secondary_symptoms(merged)

def normalize_llm_data(llm_data: Optional[dict]) -> dict:
    normalized = ensure_intake_shape(llm_data)

    duration = normalized.get("duration")
    if isinstance(duration, str):
        duration_text = duration.lower().strip()
        duration_map = {
            "từ hôm kia": "2 ngày",
            "hai ngày": "2 ngày",
            "2 hôm": "2 ngày",
            "hôm qua": "1 ngày",
            "từ hôm qua": "1 ngày",
            "vài hôm": "vài ngày",
            "mấy hôm": "vài ngày",
        }
        normalized["duration"] = duration_map.get(duration_text, duration_text)

    temperature = normalized.get("temperature")
    if temperature is not None:
        try:
            temperature_value = float(temperature)
            normalized["temperature"] = temperature_value if 34.0 <= temperature_value <= 43.0 else None
        except (ValueError, TypeError):
            normalized["temperature"] = None

    pain_score = normalized.get("pain_score")
    if pain_score is not None:
        try:
            pain_value = int(pain_score)
            normalized["pain_score"] = pain_value if 0 <= pain_value <= 10 else None
        except (ValueError, TypeError):
            normalized["pain_score"] = None

    for field in ["main_symptom", "symptom_location", "gender"]:
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = value.lower().strip()

    for list_field in ["secondary_symptoms", "red_flags", "comorbidities", "medications"]:
        clean_values = []
        for item in normalized.get(list_field, []) or []:
            if not isinstance(item, str):
                continue
            item_text = item.lower().strip()
            if item_text and item_text not in clean_values:
                clean_values.append(item_text)
        normalized[list_field] = clean_values

    normalized["ddxplus_evidences"] = normalize_evidence_codes(
        normalized.get("ddxplus_evidences", [])
    )

    main_symptom = normalized.get("main_symptom")
    normalized["secondary_symptoms"] = [
        symptom for symptom in normalized.get("secondary_symptoms", []) if symptom != main_symptom
    ]
    return normalized


def build_intake_summary(collected_data: dict) -> str:
    public_data = strip_internal_fields(collected_data)
    parts = []

    if not is_missing_value(public_data.get("age")):
        parts.append(f"tuổi: {public_data['age']}")
    if not is_missing_value(public_data.get("gender")):
        parts.append(f"giới tính: {public_data['gender']}")
    if public_data.get("is_pregnant") is True:
        parts.append("đang mang thai")
    if not is_missing_value(public_data.get("main_symptom")):
        parts.append(f"triệu chứng chính: {public_data['main_symptom']}")
    if not is_missing_value(public_data.get("duration")):
        parts.append(f"thời gian: {public_data['duration']}")
    if not is_missing_value(public_data.get("temperature")):
        parts.append(f"nhiệt độ: {public_data['temperature']} độ")
    if not is_missing_value(public_data.get("pain_score")):
        parts.append(f"điểm đau: {public_data['pain_score']}/10")
    if not is_missing_value(public_data.get("symptom_location")):
        parts.append(f"vị trí: {public_data['symptom_location']}")

    secondary_symptoms = public_data.get("secondary_symptoms", []) or []
    if secondary_symptoms:
        parts.append(f"triệu chứng kèm: {', '.join(secondary_symptoms)}")
    red_flags = public_data.get("red_flags", []) or []
    if red_flags:
        parts.append(f"dấu hiệu nguy hiểm: {', '.join(red_flags)}")
    comorbidities = public_data.get("comorbidities", []) or []
    if comorbidities:
        parts.append(f"bệnh nền: {', '.join(comorbidities)}")
    medications = public_data.get("medications", []) or []
    if medications:
        parts.append(f"thuốc đang dùng: {', '.join(medications)}")

    if not parts:
        return "Cần thêm dữ liệu về các triệu chứng."
    return "Dữ liệu đã ghi nhận: " + "; ".join(parts)


def get_missing_fields(collected_data: dict) -> dict:
    missing_required = []
    missing_optional = []

    for field in REQUIRED_FIELDS:
        if is_missing_value(collected_data.get(field)):
            missing_required.append(field)

    for field in OPTIONAL_FIELDS:
        if should_skip_optional_field(field, collected_data):
            continue
        if is_missing_value(collected_data.get(field)):
            missing_optional.append(field)

    return {"required": missing_required, "optional": missing_optional}


def choose_optional_followup(collected_data: dict, missing_optional: list) -> Optional[str]:
    main_symptom = (collected_data.get("main_symptom") or "").lower()
    secondary_symptoms = [s.lower() for s in collected_data.get("secondary_symptoms", []) or []]
    symptom_text = " ".join([main_symptom] + secondary_symptoms)

    if "red_flags" in missing_optional and (
        any(keyword in symptom_text for keyword in ["đau ngực", "khó thở", "sốt", "đau bụng", "đau đầu"])
    ):
        return "red_flags"

    if "temperature" in missing_optional and "sốt" in symptom_text:
        return "temperature"

    if "pain_score" in missing_optional and any(keyword in symptom_text for keyword in ["đau", "bụng", "ngực", "đầu", "lưng", "họng"]):
        return "pain_score"

    if "symptom_location" in missing_optional and any(keyword in symptom_text for keyword in ["đau", "bụng", "ngực", "đầu", "lưng", "họng", "ban", "ngứa"]):
        return "symptom_location"

    if "comorbidities" in missing_optional and (
        "khó thở" in symptom_text or "đau ngực" in symptom_text or "sốt" in symptom_text
    ):
        return "comorbidities"

    if "medications" in missing_optional and (
        "sốt" in symptom_text or "đau" in symptom_text or collected_data.get("comorbidities")
    ):
        return "medications"

    if "is_pregnant" in missing_optional and collected_data.get("gender") == "nữ":
        return "is_pregnant"

    return None

def extract_optional_answer(
    field_name: str,
    user_text: str,
    collected_data: dict,
    latest_rule_data: Optional[dict] = None,
    latest_llm_data: Optional[dict] = None,
) -> dict:
    updated = ensure_intake_shape(collected_data)
    latest_rule_data = ensure_intake_shape(latest_rule_data)
    latest_llm_data = ensure_intake_shape(latest_llm_data)

    if is_negative_reply(user_text):
        if field_name == "is_pregnant":
            updated["is_pregnant"] = False
        updated = mark_optional_answered(updated, field_name)
        updated["_pending_optional_field"] = None
        return updated

    list_fields = {"secondary_symptoms", "red_flags", "comorbidities", "medications"}
    if field_name in list_fields:
        current_values = updated.get(field_name, []) or []
        candidates = []

        if field_name == "secondary_symptoms":
            main_symptom = updated.get("main_symptom")
            for source in [latest_rule_data, latest_llm_data]:
                source_main = source.get("main_symptom")
                if source_main and source_main != main_symptom:
                    candidates.append(source_main)
                for symptom in source.get("secondary_symptoms", []) or []:
                    if symptom != main_symptom:
                        candidates.append(symptom)
        else:
            for source in [latest_rule_data, latest_llm_data]:
                candidates.extend(source.get(field_name, []) or [])

        updated[field_name] = merge_unique_list(current_values, candidates, exclude=updated.get("main_symptom"))
        updated = mark_optional_answered(updated, field_name)
        updated["_pending_optional_field"] = None
        return updated

    scalar_fields = {"temperature", "pain_score", "symptom_location", "age", "gender", "is_pregnant"}
    if field_name in scalar_fields:
        candidate_value = None
        if not is_missing_value(latest_llm_data.get(field_name)):
            candidate_value = latest_llm_data.get(field_name)
        elif not is_missing_value(latest_rule_data.get(field_name)):
            candidate_value = latest_rule_data.get(field_name)

        if not is_missing_value(candidate_value):
            updated[field_name] = candidate_value
        updated = mark_optional_answered(updated, field_name)
        updated["_pending_optional_field"] = None
        return updated

    updated = mark_optional_answered(updated, field_name)
    updated["_pending_optional_field"] = None
    return updated


def build_ml_input_text(collected_data: dict) -> str:
    public_data = strip_internal_fields(collected_data)
    parts: List[str] = []

    for key in ["main_symptom", "duration", "symptom_location"]:
        value = public_data.get(key)
        if not is_missing_value(value):
            parts.append(str(value))

    if not is_missing_value(public_data.get("temperature")):
        parts.append(f"sốt {public_data['temperature']} độ")

    if not is_missing_value(public_data.get("pain_score")):
        parts.append(f"đau {public_data['pain_score']}/10")

    for disease in public_data.get("comorbidities", []) or []:
        parts.append(f"bệnh nền {disease}")

    for flag in public_data.get("red_flags", []) or []:
        parts.append(f"dấu hiệu nguy hiểm {flag}")

    return " ".join(parts).strip()

def collect_symptoms(collected_data: dict) -> List[str]:
    symptoms = []

    main_symptom = collected_data.get("main_symptom")
    if not is_missing_value(main_symptom):
        if isinstance(main_symptom, str):
            for item in re.split(r"[,;/]| và ", main_symptom):
                clean = item.strip()
                if clean and clean not in symptoms:
                    symptoms.append(clean)
        else:
            symptoms.append(main_symptom)

    for symptom in collected_data.get("secondary_symptoms", []) or []:
        if symptom not in symptoms:
            symptoms.append(symptom)

    return symptoms

def build_prediction_reply(
    ml_diseases: List[dict],
    prediction_result: dict,
    department: str,
    rag_text: str,
    collected_data: Optional[dict] = None,
) -> str:
    public_data = strip_internal_fields(collected_data)
    triage_level = prediction_result.get("triage_level", "Tự theo dõi")
    triage_label = prediction_result.get("triage_label") or triage_level
    action_guidance = prediction_result.get("action_recommendation") or get_action_guidance(triage_level)
    triage_reasons = prediction_result.get("triage_reasons", []) or []

    context_lines = []
    if public_data.get("red_flags"):
        context_lines.append(f"Dấu hiệu nguy hiểm ghi nhận: {', '.join(public_data.get('red_flags', []))}")
    if not is_missing_value(public_data.get("temperature")):
        context_lines.append(f"Nhiệt độ ghi nhận: {public_data['temperature']} độ")
    if not is_missing_value(public_data.get("pain_score")):
        context_lines.append(f"Điểm đau ghi nhận: {public_data['pain_score']}/10")
    if public_data.get("comorbidities"):
        context_lines.append(f"Bệnh nền ghi nhận: {', '.join(public_data.get('comorbidities', []))}")
    if public_data.get("medications"):
        context_lines.append(f"Thuốc đang dùng ghi nhận: {', '.join(public_data.get('medications', []))}")
    if public_data.get("is_pregnant") is True:
        context_lines.append("Yếu tố nguy cơ: đang mang thai")

    if ml_diseases:
        disease_lines = []
        for item in ml_diseases[:3]:
            name = item.get("name") or item.get("label") or "Chưa rõ"
            disease_lines.append(f"- {name}")
        disease_block = "Top 3 bệnh liên quan:\n" + "\n".join(disease_lines)
    else:
        disease_block = "Top 3 bệnh liên quan:\n- Chưa đủ dữ liệu để xếp hạng bệnh cụ thể."

    reason_lines = []
    for reason in triage_reasons[:5]:
        reason_lines.append(f"- {reason}")
    if not reason_lines:
        reason_lines.append("- Dựa trên triệu chứng và dấu hiệu nguy hiểm đã khai thác.")

    context_text = ""
    if context_lines:
        context_text = "\n" + "\n".join(context_lines)
    reason_text = "\n".join(reason_lines)

    emergency_warning = ""
    if prediction_result.get("triage_priority") == 1 or "cấp cứu" in str(triage_level).lower():
        emergency_warning = (
            "Cảnh báo ưu tiên: Hệ thống phát hiện dấu hiệu nguy hiểm. "
            "Người bệnh nên đến cơ sở y tế hoặc gọi 115 ngay; phần Top bệnh bên dưới "
            "chỉ có giá trị tham khảo để sàng lọc ban đầu, không trì hoãn xử trí cấp cứu.\n\n"
        )

    return (
        "Em đã phân tích thông tin từ cuộc hội thoại.\n\n"
        f"{emergency_warning}"
        f"{disease_block}\n\n"
        f"Trạng thái sàng lọc: {triage_level} ({triage_label})\n"
        f"Khuyến nghị hành động: {action_guidance}{context_text}\n"
        f"Khoa gợi ý: {department}\n\n"
        "Vì sao hệ thống gợi ý như vậy:\n"
        f"{reason_text}\n\n"
        f"{rag_text}\n\n"
        "Lưu ý an toàn: Đây là thông tin hỗ trợ sàng lọc ban đầu, không thay thế chẩn đoán, đơn thuốc hoặc chỉ định điều trị từ bác sĩ."
    )


async def process_chat(request: ChatRequest) -> ChatResponse:
    session_repo = SessionRepository()
    prediction_repo = PredictionRepository()
    device_id = request.device_id
    session_id = request.session_id or str(uuid.uuid4())
    session_restarted = False

    existing_session = await session_repo.get_session(session_id) if request.session_id else None
    if existing_session:
        existing_device_id = existing_session.get("device_id")

        # Không cho thiết bị khác ghi tiếp vào session đã thuộc về device_id khác.
        # Nếu bị lệch device_id, backend tự tạo phiên mới thay vì dùng chung lịch sử.
        if existing_device_id and device_id and existing_device_id != device_id:
            logger.warning(
                "Session ownership mismatch; restarting session. session_id=%s",
                session_id,
            )
            session_id = str(uuid.uuid4())
            session_restarted = True
            existing_session = None
        elif existing_device_id and not device_id:
            logger.warning(
                "Session has device_id but request has no device_id; restarting session. session_id=%s",
                session_id,
            )
            session_id = str(uuid.uuid4())
            session_restarted = True
            existing_session = None
        elif not existing_device_id and device_id:
            await session_repo.attach_device_if_missing(session_id, device_id)
            existing_session["device_id"] = device_id

    if existing_session and existing_session.get("status") == "completed":
        session_id = str(uuid.uuid4())
        session_restarted = True

    await session_repo.create_session_if_not_exists(session_id, device_id=device_id)
    await session_repo.add_message(session_id, "user", request.message)

    session = await session_repo.get_session(session_id)
    messages = session.get("messages", []) if session else []

    collected_data = session.get("intake_snapshot") if session else None
    if not collected_data:
        collected_data = build_intake_from_messages(messages)
    collected_data = ensure_intake_shape(collected_data)

    latest_user_message = request.message
    pending_optional_field = collected_data.get("_pending_optional_field")

    latest_rule_data = build_intake_from_messages(
        [{"role": "user", "content": latest_user_message}],
        use_llm=False,
    )
    collected_data = merge_intake_data(collected_data, latest_rule_data)

    latest_llm_data = {}
    try:
        latest_llm_data = extract_intake_with_llm_boosted(latest_user_message, messages)
        latest_llm_data = normalize_llm_data(latest_llm_data)
        collected_data = merge_intake_data(collected_data, latest_llm_data)
    except TimeoutError as exc:
        logger.error("LLM extraction timed out: %s", exc)
    except Exception:
        logger.exception("LLM extraction failed")

    collected_data = enhance_main_symptom_with_fallback(collected_data, latest_user_message)

    if pending_optional_field:
        collected_data = extract_optional_answer(
            pending_optional_field,
            latest_user_message,
            collected_data,
            latest_rule_data=latest_rule_data,
            latest_llm_data=latest_llm_data,
        )

    await session_repo.update_intake_snapshot(session_id, collected_data)
    public_collected_data = strip_internal_fields(collected_data)

    # Red flag chỉ được dùng để nâng mức sàng lọc/cấp cứu, không còn chặn nhánh dự đoán bệnh.
    # Khi đã đủ tuổi, giới tính và triệu chứng, hệ thống vẫn chạy DDXPlus Top-K,
    # sau đó hiển thị cảnh báo cấp cứu trước phần dự đoán để người dùng không bỏ qua nguy cơ.
    emergency_result = check_emergency_triage(public_collected_data)

    missing_info = get_missing_fields(collected_data)
    missing_required = missing_info["required"]
    missing_optional = missing_info["optional"]

    if missing_required:
        missing_labels = [FIELD_LABELS.get(field, field) for field in missing_required]
        ordered_missing = [field for field in QUESTION_ORDER if field in missing_required]
        target_field = ordered_missing[0] if ordered_missing else missing_required[0]
        next_question = FIELD_QUESTIONS.get(target_field, "Anh/chị vui lòng cung cấp thêm thông tin.")
        reply = (
            f"{build_intake_summary(collected_data)}\n\n"
            f"Hiện còn thiếu dữ liệu bắt buộc trước khi dự đoán: {', '.join(missing_labels)}.\n"
            f"{next_question}"
        )
        await session_repo.add_message(session_id, "bot", reply)
        return ChatResponse(
            session_id=session_id,
            reply=reply,
            missing_fields=missing_required,
            collected_data=public_collected_data,
            next_step="continue",
            device_id=device_id,
            session_restarted=session_restarted,
        )

    optional_field_to_ask = choose_optional_followup(collected_data, missing_optional)
    if optional_field_to_ask:
        collected_data["_pending_optional_field"] = optional_field_to_ask
        await session_repo.update_intake_snapshot(session_id, collected_data)
        next_question = FIELD_QUESTIONS.get(optional_field_to_ask, "Anh/chị vui lòng cung cấp thêm thông tin.")
        reply = (
            f"{build_intake_summary(collected_data)}\n\n"
            "Em đã có đủ dữ liệu cơ bản để tiếp tục. Để đánh giá sát và an toàn hơn, em hỏi thêm một chút:\n"
            f"{next_question}"
        )
        await session_repo.add_message(session_id, "bot", reply)
        return ChatResponse(
            session_id=session_id,
            reply=reply,
            missing_fields=[],
            collected_data=strip_internal_fields(collected_data),
            next_step="continue_optional",
            device_id=device_id,
            session_restarted=session_restarted,
        )

    symptoms = collect_symptoms(collected_data)
    red_flags = collected_data.get("red_flags", []) or []
    temperature = collected_data.get("temperature")
    pain_score = collected_data.get("pain_score")
    duration = collected_data.get("duration")
    comorbidities = collected_data.get("comorbidities", []) or []
    medications = collected_data.get("medications", []) or []

    prediction_result = predict_from_symptoms(
        symptoms=symptoms,
        red_flags=red_flags,
        temperature=temperature,
        pain_score=pain_score,
        duration=duration,
        comorbidities=comorbidities,
        medications=medications,
        age=collected_data.get("age"),
        gender=collected_data.get("gender"),
        is_pregnant=collected_data.get("is_pregnant"),
        top_k=3,
    )

    ml_input_text = build_ml_input_text(collected_data)
    evidence_codes = normalize_evidence_codes(
        collected_data.get("ddxplus_evidences", []) or []
    )
    collected_data["ddxplus_evidences"] = evidence_codes
    logger.info("DDXPlus evidence codes before prediction=%r", evidence_codes)

    user_text_for_prediction = "\n".join(
        str(message.get("content") or "").strip()
        for message in messages
        if message.get("role") == "user" and str(message.get("content") or "").strip()
    ) or ml_input_text

    try:
        ml_diseases = predict_hybrid_diseases(
            user_text_for_prediction,
            symptoms=symptoms,
            top_k=3,
            age=collected_data.get("age"),
            gender=collected_data.get("gender"),
            evidence_codes=evidence_codes,
            prediction_mode="hybrid",
        )
        if not ml_diseases:
            ml_diseases = prediction_result.get("possible_diseases", []) or []
    except Exception:
        logger.exception("Hybrid disease prediction failed")
        ml_diseases = prediction_result.get("possible_diseases", []) or []

    # Dùng đủ tên Việt + canonical name để truy xuất RAG, nhưng phần giải thích
    # chỉ hiển thị tên bệnh chính cho từng Top-K để tránh lặp cùng một bệnh.
    rag_query_disease_names = []
    display_disease_names = []
    for item in ml_diseases:
        display_name = item.get("name") or item.get("canonical_name")
        if display_name and display_name not in display_disease_names:
            display_disease_names.append(display_name)
        for key in ["name", "canonical_name"]:
            value = item.get(key)
            if value and value not in rag_query_disease_names:
                rag_query_disease_names.append(value)

    if prediction_result.get("triage_priority") == 1:
        final_department = "Cấp cứu"
    else:
        inferred_department = infer_department_from_diseases(ml_diseases, symptoms=symptoms, red_flags=red_flags)
        if inferred_department == "Nội tổng quát" and prediction_result.get("department"):
            final_department = prediction_result.get("department")
        else:
            final_department = inferred_department or prediction_result.get("department") or infer_department_from_symptoms(symptoms, red_flags)

    docs = retrieve_relevant_documents(
        symptoms=symptoms,
        red_flags=red_flags,
        department=final_department,
        diseases=rag_query_disease_names,
        top_k=8,
    )
    rag_explanation = build_rag_explanation(
        docs,
        disease_names=display_disease_names,
        symptoms=symptoms,
        red_flags=red_flags,
        department=final_department,
    )

    public_collected_data = strip_internal_fields(collected_data)
    reply = build_prediction_reply(
        ml_diseases=ml_diseases,
        prediction_result=prediction_result,
        department=final_department,
        rag_text=rag_explanation,
        collected_data=public_collected_data,
    )

    prediction_payload = {
        "triage_level": prediction_result.get("triage_level"),
        "triage_priority": prediction_result.get("triage_priority"),
        "triage_label": prediction_result.get("triage_label"),
        "action_recommendation": prediction_result.get("action_recommendation"),
        "triage_reasons": prediction_result.get("triage_reasons", []),
        "department": final_department,
        "ddxplus_evidences": evidence_codes,
        "model_source": "hybrid_vn_text_ddxplus",
        "possible_diseases": [
            {
                "name": item.get("name"),
                "canonical_name": item.get("canonical_name"),
                "score": item.get("score"),
                "percent": item.get("percent"),
                "icd10": item.get("icd10"),
                "severity": item.get("severity"),
                "department": item.get("department"),
                "source": item.get("source", "ddxplus_structured_sgd"),
            }
            for item in ml_diseases[:3]
        ],
        "rag_explanation": rag_explanation,
    }

    await prediction_repo.save_prediction(
        session_id=session_id,
        collected_data=public_collected_data,
        prediction_result=prediction_payload,
    )
    await session_repo.add_message(session_id, "bot", reply)
    await session_repo.mark_session_completed(session_id)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        missing_fields=[],
        collected_data=public_collected_data,
        next_step="predicted",
        device_id=device_id,
        session_restarted=session_restarted,
    )
