import json
import logging
import re
from typing import Any, Dict, List, Optional

try:
    from google import genai
except ImportError:
    genai = None

from app.core.config import settings
from app.services.intake_service import empty_intake
from app.services.ddxplus_evidence_catalog import build_prompt_catalog, normalize_evidence_codes
from app.services.vn_evidence_candidate_service import merge_hybrid_evidences

logger = logging.getLogger(__name__)

ALLOWED_GENDER = {"nam", "nữ"}


NEGATION_WORDS = [
    "không", "khong", "chưa", "chua", "không có", "khong co", "chưa từng",
    "chưa bao giờ", "không bị", "không còn", "không thấy", "không đau",
]


RED_FLAG_TERMS = [
    "khó thở nặng", "thở không nổi", "ngộp thở", "đau ngực dữ dội", "đau ngực kéo dài",
    "lơ mơ", "lú lẫn", "mất ý thức", "ngất", "co giật", "tím tái", "yếu liệt",
    "nói khó", "méo miệng", "chảy máu không cầm", "nôn ra máu", "ho ra máu",
    "đi ngoài ra máu", "đau bụng dữ dội", "đau đầu dữ dội", "chấn thương đầu",
]


COMMON_SYMPTOM_TERMS = [
    "sốt", "sốt cao", "ớn lạnh", "ho", "ho khan", "ho có đờm", "đau họng", "rát họng",
    "sổ mũi", "nghẹt mũi", "khó thở", "thở khò khè", "đau ngực", "tức ngực",
    "hồi hộp", "tim đập nhanh", "đánh trống ngực", "vã mồ hôi", "chóng mặt", "mệt mỏi",
    "sụt cân", "tăng cân", "đau đầu", "nhìn mờ", "nói khó", "yếu liệt", "tê tay",
    "tê chân", "co giật", "ngất", "lơ mơ", "lú lẫn", "hay quên", "giảm trí nhớ",
    "khó tập trung", "mất định hướng", "đặt đồ vật sai vị trí", "không nhớ", "đau bụng",
    "đau dạ dày", "đau thượng vị", "buồn nôn", "nôn", "nôn ói", "tiêu chảy", "táo bón",
    "đầy bụng", "chướng bụng", "đi ngoài ra máu", "nôn ra máu", "đau quặn bụng", "tiểu buốt",
    "tiểu rắt", "tiểu nhiều", "tiểu ra máu", "nước tiểu đục", "dịch trắng đục", "đau lưng",
    "đau khớp", "sưng khớp", "cứng khớp", "đau cơ", "phát ban", "nổi mẩn", "ngứa",
    "da khô", "nổi mề đay", "mụn nước", "sưng ở cổ", "khó nuốt", "khàn tiếng",
    "tuyến giáp phình to", "khối u ở cổ", "run tay", "rậm lông", "ra máu bất thường",
    "chảy máu nhiều", "đau vùng chậu", "khí hư", "đau khi quan hệ", "vòng eo lớn",
    "thường xuyên đói", "khát nước", "ăn nhiều", "đổ mồ hôi", "sưng", "đau", "nhức",
]


SYMPTOM_NORMALIZATION_EXAMPLES = """
Ví dụ quy đổi bắt buộc:
- "hay đặt đồ vật sai vị trí", "không nhớ đã để đồ ở đâu" -> main_symptom: "hay quên, giảm trí nhớ, đặt đồ vật sai vị trí"
- "sưng ở cổ, khó nuốt và khàn tiếng" -> main_symptom: "sưng ở cổ, khó nuốt, khàn tiếng"
- "tiểu buốt, tiểu rát và dịch trắng đục" -> main_symptom: "tiểu buốt, tiểu rát, dịch trắng đục"
- "hồi hộp, tim đập nhanh, run tay và sụt cân" -> main_symptom: "hồi hộp, tim đập nhanh, run tay, sụt cân"
- "vòng eo lớn và thường xuyên đói bụng" -> main_symptom: "vòng eo lớn, thường xuyên đói bụng"
- "sưng cổ", "cục ở cổ", "bướu ở cổ", "tuyến giáp phình to" -> main_symptom: "sưng ở cổ, tuyến giáp phình to"
- "ngứa và da khô" -> main_symptom: "ngứa, da khô"
- "đau quặn bụng, tiêu chảy ra máu và sút cân" -> main_symptom: "đau quặn bụng, tiêu chảy ra máu, sút cân"
- "ra máu nhiều bất thường sau sinh" -> main_symptom: "ra máu nhiều bất thường sau sinh"
- "nhức mình", "đau ê ẩm", "mỏi rã rời" -> main_symptom: "đau nhức toàn thân"
- "xây xẩm", "choáng váng", "lảo đảo", "mắt tối sầm" -> main_symptom: "chóng mặt"
- "ói", "mửa" -> main_symptom: "nôn"
- "đi ngoài", "tào tháo rượt" -> main_symptom: "tiêu chảy"
- "thở không nổi", "hụt hơi", "ngộp thở" -> main_symptom: "khó thở"
""".strip()


def normalize_text(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split()).lower()
    return text or None


def normalize_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    results = []
    for item in values:
        text = normalize_text(item)
        if text and text not in results:
            results.append(text)
    return results


def normalize_gender(value: Any) -> Optional[str]:
    text = normalize_text(value)
    if text in ALLOWED_GENDER:
        return text
    if text in {"nu", "nữ giới", "giới tính nữ", "con gái", "phụ nữ"}:
        return "nữ"
    if text in {"nam giới", "giới tính nam", "con trai", "đàn ông"}:
        return "nam"
    return None


def normalize_age(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        age = int(value)
        if 0 < age <= 120:
            return age
    except (ValueError, TypeError):
        return None
    return None


def normalize_temperature(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", ".").strip()
        temp = float(value)
        if 34.0 <= temp <= 43.0:
            return temp
    except (ValueError, TypeError):
        return None
    return None


def normalize_pain_score(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        score = int(value)
        if 0 <= score <= 10:
            return score
    except (TypeError, ValueError):
        return None
    return None


def normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = normalize_text(value)
    if text in {"true", "co", "có", "yes", "đúng", "mang thai", "có thai", "đang bầu", "có bầu"}:
        return True
    if text in {"false", "khong", "không", "no", "không có", "không mang thai", "không bầu"}:
        return False
    return None


def split_symptom_text(value: Optional[str]) -> List[str]:
    if not isinstance(value, str):
        return []
    text = normalize_text(value)
    if not text:
        return []
    parts = re.split(r"[,;/]|\s+và\s+|\s+kèm\s+|\s+cùng\s+", text)
    results = []
    for part in parts:
        clean = part.strip(" ,.;:-")
        if clean and clean not in results:
            results.append(clean)
    return results


def is_negated_term(text: str, term: str) -> bool:
    if not text or not term or term not in text:
        return False
    index = text.find(term)
    window = text[max(0, index - 28): index]
    return any(word in window for word in NEGATION_WORDS)


def extract_keywords_from_text(user_text: str) -> List[str]:
    text = normalize_text(user_text) or ""
    found = []
    for term in COMMON_SYMPTOM_TERMS:
        if term in text and not is_negated_term(text, term) and term not in found:
            found.append(term)
    return found


def fallback_main_symptom_from_text(user_text: str) -> Optional[str]:
    text = normalize_text(user_text)
    if not text:
        return None

    patterns = [
        r"các triệu chứng như\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
        r"triệu chứng như\s+(.+?)(?:\.|,?\s*tôi có thể|,?\s*em có thể|$)",
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
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            value = re.sub(r"\b(có thể là bệnh gì|bị bệnh gì|là bệnh gì).*$", "", value)
            value = re.sub(r"\b(trong|khoảng|kéo dài|đã kéo dài|từ)\s+\d+\s*(giờ|ngày|tuần|tháng|năm).*$", "", value)
            value = value.strip(" ,.;:-")
            if len(value) >= 3:
                return value[:180]

    keywords = extract_keywords_from_text(text)
    if keywords:
        return ", ".join(keywords[:15])

    health_words = ["đau", "sốt", "ho", "khó thở", "ngứa", "sưng", "nôn", "tiêu chảy", "tiểu", "chảy máu", "mệt", "quên", "khó nuốt", "khàn tiếng"]
    if any(word in text for word in health_words) and len(text) >= 10:
        value = re.sub(r"\b(tôi|em|mình|cháu)\s+có\s+thể\s+.*$", "", text)
        return value.strip(" ,.;:-")[:180]

    return None


def combine_symptoms(main_symptom: Optional[str], secondary_symptoms: List[str]) -> Optional[str]:
    parts = []
    for item in split_symptom_text(main_symptom):
        if item not in parts:
            parts.append(item)
    for item in secondary_symptoms or []:
        for sub_item in split_symptom_text(item):
            if sub_item not in parts:
                parts.append(sub_item)
    if not parts:
        return None
    return ", ".join(parts[:20])


def normalize_llm_result(data: Dict[str, Any], user_text: Optional[str] = None) -> Dict[str, Any]:
    result = empty_intake()
    result["age"] = normalize_age(data.get("age"))
    result["gender"] = normalize_gender(data.get("gender"))
    result["duration"] = normalize_text(data.get("duration"))
    result["temperature"] = normalize_temperature(data.get("temperature"))
    result["pain_score"] = normalize_pain_score(data.get("pain_score"))
    result["symptom_location"] = normalize_text(data.get("symptom_location"))
    result["is_pregnant"] = normalize_bool(data.get("is_pregnant"))
    llm_evidence_codes = normalize_evidence_codes(data.get("ddxplus_evidences"))
    hybrid_evidence = merge_hybrid_evidences(user_text or "", llm_evidence_codes)
    result["ddxplus_evidences"] = hybrid_evidence["positive"]
    result["ddxplus_negative_evidences"] = hybrid_evidence["negative"]
    result["ddxplus_rule_evidences"] = hybrid_evidence["rule_positive"]
    result["ddxplus_evidence_matches"] = hybrid_evidence["matched_aliases"]

    raw_main_symptom = normalize_text(data.get("main_symptom"))
    secondary_symptoms = normalize_list(data.get("secondary_symptoms"))
    fallback_main = fallback_main_symptom_from_text(user_text or "")

    if not raw_main_symptom:
        raw_main_symptom = fallback_main
    elif fallback_main:
        extracted_keywords = extract_keywords_from_text(user_text or "")
        for keyword in extracted_keywords:
            if keyword not in secondary_symptoms and keyword not in split_symptom_text(raw_main_symptom):
                secondary_symptoms.append(keyword)

    result["main_symptom"] = combine_symptoms(raw_main_symptom, secondary_symptoms)
    result["secondary_symptoms"] = []
    result["red_flags"] = normalize_list(data.get("red_flags"))
    result["comorbidities"] = normalize_list(data.get("comorbidities"))
    result["medications"] = normalize_list(data.get("medications"))

    if user_text:
        text = normalize_text(user_text) or ""
        clean_red_flags = []
        for flag in result["red_flags"]:
            if flag and not is_negated_term(text, flag):
                clean_red_flags.append(flag)
        for flag in RED_FLAG_TERMS:
            if flag in text and not is_negated_term(text, flag) and flag not in clean_red_flags:
                clean_red_flags.append(flag)
        result["red_flags"] = clean_red_flags

    return result


def extract_json_text(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        raise ValueError("LLM response is not a string")
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()
    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0).strip()
    return text


def get_client():
    api_key = getattr(settings, "gemini_api_key", None)
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)


def get_model_name() -> str:
    return (
        getattr(settings, "gemini_extractor_model", None)
        or getattr(settings, "gemini_model", None)
        or "gemini-3.1-flash-lite"
    )


def build_prompt(user_text: str) -> str:
    ddx_catalog = build_prompt_catalog()
    return f"""
Bạn là bộ trích xuất intake y tế tiếng Việt cho chatbot sàng lọc ban đầu.

Mục tiêu quan trọng nhất:
- Phải nhận diện main_symptom thật mạnh.
- Nếu người dùng mô tả nhiều triệu chứng trong cùng một câu, hãy gom tất cả triệu chứng thật sự vào main_symptom, ngăn cách bằng dấu phẩy.
- Không hỏi lại trong JSON. Không chẩn đoán bệnh. Không giải thích.

Nhiệm vụ:
1. Hiểu câu nói tự nhiên tiếng Việt, gồm tiếng địa phương, câu dài, câu hỏi "tôi có thể bị bệnh gì".
2. Quy đổi mô tả đời thường thành cụm triệu chứng y khoa phổ thông.
3. main_symptom không chỉ là 1 từ. main_symptom phải là chuỗi triệu chứng đầy đủ nhất có trong câu.
4. secondary_symptoms trả về [] để tránh tách rời triệu chứng phụ.
5. Nếu câu có mẫu "các triệu chứng như...", "tôi bị...", "tôi đang có...", hãy lấy phần mô tả sau đó làm main_symptom sau khi làm sạch.
6. Nếu không rõ triệu chứng nào chính, lấy toàn bộ các biểu hiện sức khỏe trong câu làm main_symptom.
7. Chỉ lấy dữ liệu có trong câu người dùng, không tự bịa tuổi, giới tính, bệnh nền, thuốc, thai kỳ.
8. Ánh xạ các biểu hiện thật sự có mặt sang mã DDXPlus trong danh mục bên dưới. Chỉ chọn mã chắc chắn phù hợp.

{SYMPTOM_NORMALIZATION_EXAMPLES}

Schema JSON bắt buộc:
{{
  "age": integer hoặc null,
  "gender": "nam" | "nữ" | null,
  "main_symptom": string hoặc null,
  "duration": string hoặc null,
  "temperature": number hoặc null,
  "pain_score": integer từ 0 đến 10 hoặc null,
  "symptom_location": string hoặc null,
  "secondary_symptoms": [],
  "red_flags": array string,
  "comorbidities": array string,
  "medications": array string,
  "is_pregnant": true | false | null,
  "ddxplus_evidences": array string
}}


Quy tắc mã bằng chứng DDXPlus:
- Chỉ trả về mã xuất hiện trong danh mục. Không tự tạo mã mới.
- Chỉ đưa bằng chứng đang có hoặc tiền sử được người dùng xác nhận; bỏ các biểu hiện bị phủ định.
- Với bằng chứng B, dùng dạng E_x. Với bằng chứng C/M, bắt buộc dùng E_x_@_giá_trị chính xác.
- Nếu chưa đủ chắc chắn để chọn giá trị C/M thì không đưa mã đó vào danh sách.
- Nếu câu có ít nhất một triệu chứng hoặc tiền sử rõ ràng, hãy cố gắng trả ít nhất một mã hợp lệ. Chỉ trả [] khi thật sự không có bằng chứng y khoa nào để ánh xạ.

Danh mục bằng chứng DDXPlus (tiếng Anh chuẩn):
{ddx_catalog}

Quy tắc phủ định:
- Nếu người dùng nói "không khó thở", "không đau ngực", "chưa từng ngất", "không co giật" thì không đưa các cụm đó vào main_symptom hoặc red_flags.
- Nếu người dùng phủ định red flag, bỏ red flag đó.

Quy tắc red_flags:
- Chỉ đưa vào red_flags khi là dấu hiệu nguy hiểm đang có thật sự.
- Ví dụ red_flags: khó thở nặng, đau ngực dữ dội/kéo dài, lơ mơ/lú lẫn, ngất, co giật, tím tái, yếu liệt/nói khó/méo miệng, chảy máu không cầm, nôn ra máu, ho ra máu, đi ngoài ra máu, đau bụng dữ dội, đau đầu dữ dội.

Quy tắc số liệu:
- temperature chỉ điền khi có nhiệt độ hợp lệ như 38.5, 39 độ.
- pain_score chỉ điền khi người dùng nói rõ số 0-10 như "đau 7/10", "điểm đau 8".
- duration chỉ điền khi có thời gian như 2 ngày, vài tuần, 3 tháng, từ hôm qua.

Câu người dùng:
{user_text}

Chỉ trả về JSON hợp lệ.
""".strip()


def build_rescue_prompt(user_text: str) -> str:
    return f"""
Hãy trích xuất lại main_symptom từ câu người dùng. Trả về JSON hợp lệ, không giải thích.

Yêu cầu:
- main_symptom bắt buộc phải có nếu câu có bất kỳ biểu hiện sức khỏe nào.
- Gom tất cả triệu chứng trong câu vào main_symptom bằng dấu phẩy.
- Không chẩn đoán bệnh.
- Không đưa triệu chứng bị phủ định vào main_symptom.
- secondary_symptoms luôn là [].

Ví dụ:
"Tôi hay đặt đồ vật sai vị trí và không thể nhớ đã để chúng ở đâu" -> "hay quên, giảm trí nhớ, đặt đồ vật sai vị trí"
"Tôi hiện đang có các triệu chứng như sưng ở cổ, khó nuốt và khàn tiếng" -> "sưng ở cổ, khó nuốt, khàn tiếng"
"Tôi hiện đang có số đo vòng eo lớn và thường xuyên cảm thấy đói bụng" -> "vòng eo lớn, thường xuyên đói bụng"

Schema:
{{
  "age": null,
  "gender": null,
  "main_symptom": string hoặc null,
  "duration": string hoặc null,
  "temperature": number hoặc null,
  "pain_score": integer hoặc null,
  "symptom_location": string hoặc null,
  "secondary_symptoms": [],
  "red_flags": [],
  "comorbidities": [],
  "medications": [],
  "is_pregnant": null,
  "ddxplus_evidences": []
}}

Câu người dùng:
{user_text}
""".strip()


def build_evidence_rescue_prompt(user_text: str) -> str:
    """Build a focused second-pass prompt when intake extraction has no evidence."""
    ddx_catalog = build_prompt_catalog()
    return f"""
Bạn là bộ ánh xạ triệu chứng tiếng Việt sang mã bằng chứng DDXPlus.

Nhiệm vụ duy nhất:
- Đọc câu người dùng và chọn các mã DDXPlus phù hợp nhất từ danh mục bên dưới.
- Không chẩn đoán bệnh. Không giải thích. Không tự tạo mã.
- Bỏ mọi triệu chứng bị phủ định như "không khó thở", "không đau ngực".
- Với evidence loại B, trả dạng E_x.
- Với evidence loại C hoặc M, bắt buộc trả dạng E_x_@_V_x hoặc E_x_@_giá_trị đúng danh mục.
- Nếu câu có triệu chứng rõ ràng, cố gắng chọn ít nhất một mã hợp lệ.
- Chỉ trả JSON hợp lệ.

Schema:
{{
  "ddxplus_evidences": ["E_x", "E_x_@_V_x"]
}}

Danh mục bằng chứng DDXPlus:
{ddx_catalog}

Câu người dùng:
{user_text}
""".strip()


def call_gemini_json(client, prompt: str) -> Dict[str, Any]:
    response = client.models.generate_content(model=get_model_name(), contents=prompt)
    raw_text = (response.text or "").strip()
    json_text = extract_json_text(raw_text)
    data = json.loads(json_text)
    if not isinstance(data, dict):
        return {}
    return data


def extract_intake_rule_only(user_text: str) -> dict:
    """Deterministic fallback used for benchmarks and Gemini failures."""
    result = empty_intake()
    text = normalize_text(user_text) or ""
    result["main_symptom"] = fallback_main_symptom_from_text(user_text)
    result["secondary_symptoms"] = []

    age_match = re.search(r"\b(\d{1,3})\s*tuổi\b", text)
    if age_match:
        result["age"] = normalize_age(age_match.group(1))
    if re.search(r"\b(nữ|phụ nữ|nữ giới|cô gái|bé gái)\b", text):
        result["gender"] = "nữ"
    elif re.search(r"\b(nam|đàn ông|nam giới|cậu bé|bé trai)\b", text):
        result["gender"] = "nam"

    temperature_match = re.search(r"\b(3[4-9]|4[0-3])(?:[,.](\d))?\s*(?:độ|°c|c)\b", text)
    if temperature_match:
        raw = temperature_match.group(1)
        if temperature_match.group(2):
            raw += "." + temperature_match.group(2)
        result["temperature"] = normalize_temperature(raw)

    pain_match = re.search(r"(?:đau|điểm đau)\s*(\d{1,2})\s*/\s*10", text)
    if pain_match:
        result["pain_score"] = normalize_pain_score(pain_match.group(1))

    result["red_flags"] = [
        flag for flag in RED_FLAG_TERMS
        if flag in text and not is_negated_term(text, flag)
    ]
    hybrid_evidence = merge_hybrid_evidences(user_text, [])
    result["ddxplus_evidences"] = hybrid_evidence["positive"]
    result["ddxplus_negative_evidences"] = hybrid_evidence["negative"]
    result["ddxplus_rule_evidences"] = hybrid_evidence["rule_positive"]
    result["ddxplus_evidence_matches"] = hybrid_evidence["matched_aliases"]
    return result


def extract_intake_with_llm(user_text: str) -> dict:
    if not isinstance(user_text, str) or not user_text.strip():
        return empty_intake()

    client = get_client()
    if client is None:
        return extract_intake_rule_only(user_text)

    try:
        data = call_gemini_json(client, build_prompt(user_text))
        logger.info(
            "Gemini raw ddxplus_evidences=%r",
            data.get("ddxplus_evidences"),
        )
        result = normalize_llm_result(data, user_text=user_text)
        logger.info(
            "Normalized ddxplus_evidences=%r",
            result.get("ddxplus_evidences"),
        )

        if not result.get("main_symptom"):
            rescue_data = call_gemini_json(client, build_rescue_prompt(user_text))
            rescue_result = normalize_llm_result(rescue_data, user_text=user_text)
            if rescue_result.get("main_symptom"):
                result["main_symptom"] = rescue_result.get("main_symptom")
                result["secondary_symptoms"] = []

        # A successful Gemini request may still return an empty evidence list.
        # Run one focused mapping pass before giving up.
        if not result.get("ddxplus_evidences"):
            evidence_data = call_gemini_json(
                client,
                build_evidence_rescue_prompt(user_text),
            )
            rescued_codes = normalize_evidence_codes(
                evidence_data.get("ddxplus_evidences")
            )
            logger.info("Rescued ddxplus_evidences=%r", rescued_codes)
            if rescued_codes:
                merged_rescue = merge_hybrid_evidences(
                    user_text,
                    list(result.get("ddxplus_evidences", []) or []) + rescued_codes,
                )
                result["ddxplus_evidences"] = merged_rescue["positive"]
                result["ddxplus_negative_evidences"] = merged_rescue["negative"]
                result["ddxplus_rule_evidences"] = merged_rescue["rule_positive"]
                result["ddxplus_evidence_matches"] = merged_rescue["matched_aliases"]

        if not result.get("main_symptom"):
            result["main_symptom"] = fallback_main_symptom_from_text(user_text)
            result["secondary_symptoms"] = []

        return result
    except TimeoutError as exc:
        logger.error("Gemini extraction timed out: %s", exc)
        return extract_intake_rule_only(user_text)
    except Exception:
        logger.exception("LLM extraction failed")
        return extract_intake_rule_only(user_text)
