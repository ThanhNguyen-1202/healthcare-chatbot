from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.services.rag_translation_service import translate_rag_doc_to_vietnamese


DISEASE_TRANSLATION_ALIASES = {
    "bàn chân đái tháo đường": [
        "diabetic foot",
        "diabetic foot problems",
        "diabetes foot",
        "bàn chân tiểu đường",
        "nhiễm trùng bàn chân tiểu đường",
    ],
    "nhiễm trùng bàn chân tiểu đường": [
        "diabetic foot",
        "diabetic foot infection",
        "diabetic foot problems",
        "bàn chân đái tháo đường",
        "bàn chân tiểu đường",
    ],
    "đái tháo đường": ["diabetes", "diabetes mellitus", "tiểu đường"],
    "biến chứng đái tháo đường": ["diabetes complications", "diabetic complications"],
    "viêm da cơ địa": ["atopic dermatitis", "eczema", "chàm"],
    "sa sút trí tuệ": ["dementia", "memory loss", "alzheimer", "alzheimer's disease"],
    "hẹp động mạch cảnh": ["carotid artery disease", "carotid stenosis"],
    "bệnh moyamoya": ["moyamoya disease"],
    "viêm xoang": ["sinusitis", "sinus infection"],
    "cảm cúm": ["flu", "influenza", "common cold"],
    "đau đầu căng thẳng": ["tension headache", "headache"],
}


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().strip().split())


def tokenize_vi(text: str) -> Set[str]:
    text = normalize_text(text)
    return {
        token
        for token in re.findall(r"[\wÀ-ỹ]+", text, flags=re.UNICODE)
        if len(token) >= 2
    }


def normalize_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    results: List[str] = []
    for item in values:
        if isinstance(item, str):
            text = item.strip()
            if text and text not in results:
                results.append(text)
    return results


def looks_like_untranslated_english(
    original_title: str,
    original_content: str,
    title_vi: str,
    content_vi: str,
) -> bool:
    original_title_norm = (original_title or "").strip().lower()
    original_content_norm = (original_content or "").strip().lower()
    title_vi_norm = (title_vi or "").strip().lower()
    content_vi_norm = (content_vi or "").strip().lower()

    if not title_vi_norm and not content_vi_norm:
        return True
    if title_vi_norm == original_title_norm and content_vi_norm == original_content_norm:
        return True

    english_markers = [
        "what is",
        "symptoms",
        "causes",
        "treatment",
        "diagnosis",
        "infection",
        "disease",
        "lungs",
        "breathing",
        "pneumonia",
        "call your health care provider",
    ]
    combined = f"{title_vi_norm} {content_vi_norm}"
    return any(marker in combined for marker in english_markers)


def build_vietnamese_fallback(doc: Dict[str, Any]) -> Tuple[str, str]:
    disease_names = normalize_list(doc.get("disease_names", []))
    symptoms = normalize_list(doc.get("symptoms", []))
    red_flags = normalize_list(doc.get("red_flags", []))
    department = (doc.get("department") or "").strip()
    title = (doc.get("title") or "Tài liệu tham khảo").strip()
    display_title = disease_names[0] if disease_names else title

    sentences = []
    if disease_names:
        if len(disease_names) == 1:
            sentences.append(f"Tài liệu này liên quan đến bệnh {disease_names[0]}.")
        else:
            sentences.append(f"Tài liệu này liên quan đến các bệnh như {', '.join(disease_names[:3])}.")
    else:
        sentences.append(f"Tài liệu này liên quan đến chủ đề {title}.")

    if symptoms:
        sentences.append(f"Các biểu hiện thường liên quan gồm: {', '.join(symptoms[:6])}.")
    if department:
        sentences.append(f"Khoa phù hợp để thăm khám ban đầu là {department}.")
    if red_flags:
        sentences.append(f"Nếu có dấu hiệu như {', '.join(red_flags[:5])}, người bệnh nên đi khám sớm.")
    if len(sentences) == 1:
        sentences.append("Người bệnh nên theo dõi thêm và đi khám nếu triệu chứng kéo dài hoặc tăng nặng.")
    return display_title, " ".join(sentences)


def shorten_text(text: str, max_length: int = 900) -> str:
    """Return a complete, readable excerpt without trailing ellipsis.

    Người dùng yêu cầu phần gợi ý/tài liệu không được bị cắt dạng "...".
    Hàm này giữ các câu hoàn chỉnh trong giới hạn độ dài; nếu nguồn quá dài,
    chỉ lấy các câu đầu đã hoàn chỉnh và vẫn kết thúc bằng dấu câu.
    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    if len(text) <= max_length:
        return ensure_sentence_end(text)

    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    selected: List[str] = []
    total = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        next_total = total + len(sentence) + (1 if selected else 0)
        if selected and next_total > max_length:
            break
        if not selected and len(sentence) > max_length:
            break
        selected.append(sentence)
        total = next_total

    if selected:
        return ensure_sentence_end(" ".join(selected))

    # Fallback khi văn bản không có dấu câu rõ ràng: cắt theo từ, không dùng dấu ba chấm.
    words = text[:max_length].rsplit(" ", 1)[0].strip()
    return ensure_sentence_end(words)


def ensure_sentence_end(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    if text[-1] in ".!?…。！？":
        # Chuẩn hóa dấu ba chấm thành dấu chấm để không còn câu bị lửng.
        return text.rstrip("…").rstrip(".") + "." if text.endswith("…") else text
    return text + "."


def _prepare_doc_for_display(doc: Dict[str, Any]) -> Tuple[str, str, str]:
    title = (doc.get("title") or "Tài liệu tham khảo").strip()
    content = (doc.get("content") or "").strip()
    source = (doc.get("source") or "knowledge_base_internal").strip()

    if not content:
        return title, "", source

    display_title = title
    display_content = content

    if source.lower() == "medlineplus":
        translated = translate_rag_doc_to_vietnamese(title, content)
        translated_title = (translated.get("title_vi") or "").strip()
        translated_content = (translated.get("content_vi") or "").strip()

        if looks_like_untranslated_english(title, content, translated_title, translated_content):
            display_title, display_content = build_vietnamese_fallback(doc)
        else:
            display_title = translated_title or title
            display_content = translated_content or content

    return display_title, shorten_text(display_content), source


def _doc_text_for_matching(doc: Dict[str, Any]) -> str:
    parts = [
        doc.get("title", ""),
        doc.get("content", ""),
        doc.get("department", ""),
        " ".join(doc.get("disease_names", []) or []),
        " ".join(doc.get("disease_aliases", []) or []),
        " ".join(doc.get("symptoms", []) or []),
        " ".join(doc.get("red_flags", []) or []),
    ]
    return normalize_text(" ".join(str(part) for part in parts if part))


def _disease_aliases(disease_name: str) -> Set[str]:
    disease_norm = normalize_text(disease_name)
    aliases = {disease_norm} if disease_norm else set()

    for key, values in DISEASE_TRANSLATION_ALIASES.items():
        key_norm = normalize_text(key)
        value_norms = {normalize_text(item) for item in values}
        if disease_norm == key_norm or disease_norm in value_norms or key_norm in disease_norm:
            aliases.add(key_norm)
            aliases.update(value_norms)
    return {item for item in aliases if item}


def _matched_terms_in_doc(doc: Dict[str, Any], terms: List[str]) -> List[str]:
    doc_text = _doc_text_for_matching(doc)
    matched: List[str] = []
    for term in normalize_list(terms):
        term_norm = normalize_text(term)
        if term_norm and term_norm in doc_text and term not in matched:
            matched.append(term)
    return matched


def _combined_doc_score(
    doc: Dict[str, Any],
    disease_name: str,
    symptoms: List[str],
    red_flags: List[str],
) -> Tuple[float, List[str]]:
    """Score one RAG doc by BOTH predicted disease and entered symptoms.\n\n    Không còn bắt buộc tài liệu phải khớp 100% theo tên bệnh. Một tài liệu có thể\n    được dùng để giải thích nếu nó khớp theo tên bệnh, alias Việt-Anh, hoặc khớp\n    nhiều triệu chứng người dùng nhập.\n    """
    doc_text = _doc_text_for_matching(doc)
    doc_tokens = tokenize_vi(doc_text)
    score = 0.0
    reasons: List[str] = []

    # 1) Tên bệnh / alias bệnh.
    for alias in _disease_aliases(disease_name):
        if alias and alias in doc_text:
            score += 10
            reasons.append(f"tài liệu có nhắc đến bệnh/alias '{alias}'")
            break

    # 2) Triệu chứng chính và triệu chứng phụ.
    symptom_matches = _matched_terms_in_doc(doc, symptoms)
    if symptom_matches:
        score += len(symptom_matches) * 5
        reasons.append("khớp triệu chứng: " + ", ".join(symptom_matches[:5]))

    # 3) Red flag nếu có.
    red_flag_matches = _matched_terms_in_doc(doc, red_flags)
    if red_flag_matches:
        score += len(red_flag_matches) * 7
        reasons.append("khớp dấu hiệu nguy hiểm: " + ", ".join(red_flag_matches[:4]))

    # 4) Khớp từ khóa rời giữa tên bệnh và tài liệu, dùng nhẹ để hỗ trợ khi tên Việt-Anh lệch.
    disease_tokens = tokenize_vi(disease_name)
    overlap = disease_tokens.intersection(doc_tokens)
    if overlap:
        score += min(len(overlap), 4) * 1.5

    # 5) Điểm retrieval có sẵn từ retriever.
    try:
        score += min(float(doc.get("retrieval_score", 0) or 0), 20) * 0.15
    except Exception:
        pass

    return score, reasons


def _find_best_doc_for_disease_and_symptoms(
    docs: List[Dict[str, Any]],
    disease_name: str,
    symptoms: List[str],
    red_flags: List[str],
) -> Tuple[Optional[Dict[str, Any]], List[str], float]:
    best_doc: Optional[Dict[str, Any]] = None
    best_reasons: List[str] = []
    best_score = 0.0

    for doc in docs or []:
        score, reasons = _combined_doc_score(doc, disease_name, symptoms, red_flags)
        if score > best_score:
            best_doc = doc
            best_reasons = reasons
            best_score = score

    # Ngưỡng thấp để cho phép giải thích theo triệu chứng khi tên bệnh Việt-Anh chưa map đủ.
    if best_score >= 3:
        return best_doc, best_reasons, best_score
    return None, [], best_score


def _symptom_text(symptoms: Optional[List[str]], red_flags: Optional[List[str]]) -> str:
    symptoms = normalize_list(symptoms or [])
    red_flags = normalize_list(red_flags or [])
    parts = []
    if symptoms:
        parts.append(", ".join(symptoms[:8]))
    if red_flags:
        parts.append("dấu hiệu nguy hiểm: " + ", ".join(red_flags[:5]))
    return "; ".join(parts) if parts else "triệu chứng đã nhập"


def _safe_reason_text(reasons: List[str]) -> str:
    if not reasons:
        return "được đối chiếu theo cả tên bệnh nghi ngờ và nhóm triệu chứng người dùng nhập"
    return "; ".join(reasons[:3])


def build_rag_explanation(
    docs: List[Dict[str, Any]],
    disease_names: List[str] = None,
    symptoms: List[str] = None,
    red_flags: List[str] = None,
    department: str = None,
) -> str:
    """Build explanation by combining predicted diseases + user symptoms + RAG docs.\n\n    Phiên bản này không chỉ lấy đúng tên bệnh để giải thích. Nó chọn tài liệu RAG\n    theo điểm khớp kết hợp: tên bệnh/alias + triệu chứng + red flag + điểm retriever.\n    """
    disease_names = normalize_list(disease_names or [])[:3]
    symptoms = normalize_list(symptoms or [])
    red_flags = normalize_list(red_flags or [])
    symptoms_text = _symptom_text(symptoms, red_flags)
    department_text = (department or "").strip()

    lines = [
        "Giải thích theo bệnh nghi ngờ và triệu chứng:",
        f"- Triệu chứng dùng để đối chiếu: {symptoms_text}.",
        "- Hệ thống giải thích bằng cách kết hợp tên bệnh Top 3 với triệu chứng đã nhập và tài liệu liên quan, đây không phải chẩn đoán chắc chắn.",
    ]
    if department_text:
        lines.append(f"- Khoa gợi ý để kiểm tra ban đầu: {department_text}.")

    source_lines: List[str] = []
    used_evidence_ids: Set[str] = set()

    if not disease_names:
        if not docs:
            lines.append("- Chưa có đủ bệnh nghi ngờ hoặc tài liệu để giải thích chi tiết.")
        else:
            lines.append("- Chưa có danh sách bệnh Top 3 rõ ràng, nên hệ thống chỉ giải thích theo triệu chứng và tài liệu gần nhất.")
            for idx, doc in enumerate(docs[:3], start=1):
                title, content, source = _prepare_doc_for_display(doc)
                if content:
                    lines.append(f"- [{idx}] {title}: {content}")
                    source_lines.append(f"[{idx}] {title} - {source or 'knowledge_base_internal'} ({doc.get('evidence_id') or f'KB{idx}'})")
    else:
        for idx, disease in enumerate(disease_names, start=1):
            matched_doc, reasons, _score = _find_best_doc_for_disease_and_symptoms(
                docs=docs,
                disease_name=disease,
                symptoms=symptoms,
                red_flags=red_flags,
            )

            if matched_doc:
                title, content, source = _prepare_doc_for_display(matched_doc)
                reason_text = _safe_reason_text(reasons)
                if content:
                    lines.append(f"- {disease}: {ensure_sentence_end(content)}")
                else:
                    lines.append(f"- {disease}: Tài liệu tham khảo có nhắc đến bệnh này nhưng chưa có nội dung mô tả đủ rõ để giải thích. Người bệnh nên được bác sĩ kiểm chứng khi thăm khám.")
                evidence_id = str(matched_doc.get("evidence_id") or f"KB{idx}")
                if evidence_id not in used_evidence_ids:
                    used_evidence_ids.add(evidence_id)
                    source_lines.append(f"[{len(source_lines) + 1}] {title} - {source or 'knowledge_base_internal'} ({evidence_id})")
            else:
                lines.append(
                    f"- {disease}: được mô hình dự đoán từ dữ liệu triệu chứng, nhưng trong các tài liệu đã truy xuất "
                    "chưa có đoạn đủ khớp với cả tên bệnh và triệu chứng. Mục này cần bác sĩ kiểm chứng khi thăm khám."
                )

    if source_lines:
        lines.append("")
        lines.append("Nguồn được dùng để đối chiếu:")
        lines.extend(source_lines)
    else:
        lines.append("")
        lines.append("Nguồn: Chưa có tài liệu đủ khớp để giải thích chi tiết theo cả bệnh và triệu chứng.")

    return "\n".join(lines)
