from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

KNOWLEDGE_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "processed"
    / "rag_knowledge_base_merged.jsonl"
)

DDXPLUS_DISEASE_SUPPLEMENT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "processed"
    / "ddxplus_49_disease_rag_supplement.jsonl"
)

ALIAS_PATH = Path(__file__).resolve().parent.parent / "mappings" / "disease_aliases.json"


DEFAULT_DISEASE_TRANSLATION_ALIASES = {
   
    "bàn chân đái tháo đường": [
        "diabetic foot", "diabetic foot problems", "diabetes foot",
        "bàn chân tiểu đường", "nhiễm trùng bàn chân tiểu đường"
    ],
    "nhiễm trùng bàn chân tiểu đường": [
        "diabetic foot", "diabetic foot infection", "diabetic foot problems",
        "bàn chân đái tháo đường", "bàn chân tiểu đường"
    ],
    "đái tháo đường": ["diabetes", "diabetes mellitus", "diabetes type 1", "diabetes type 2", "tiểu đường"],
    "biến chứng đái tháo đường": ["diabetes complications", "diabetic complications", "diabetic foot", "diabetic eye problems", "diabetic kidney problems"],
    "bệnh thận do đái tháo đường": ["diabetic kidney problems", "diabetic nephropathy"],
    "bệnh mắt do đái tháo đường": ["diabetic eye problems", "diabetic retinopathy"],
    "bệnh thần kinh do đái tháo đường": ["diabetic nerve problems", "diabetic neuropathy"],
    "viêm da cơ địa": ["atopic dermatitis", "eczema", "chàm"],
    "sa sút trí tuệ": ["dementia", "memory loss", "alzheimer", "alzheimer's disease"],
    "hẹp động mạch cảnh": ["carotid artery disease", "carotid stenosis"],
    "bệnh moyamoya": ["moyamoya disease"],
    "viêm xoang": ["sinusitis", "sinus infection"],
    "cảm cúm": ["flu", "influenza", "common cold"],
    "đau đầu căng thẳng": ["tension headache", "headache"],
}

_knowledge_cache: Optional[List[Dict[str, Any]]] = None
_alias_cache: Optional[Dict[str, List[str]]] = None
_index_cache: Optional[Dict[str, Any]] = None
_vector_payload_cache: Optional[List[Dict[str, Any]]] = None
_cache_lock = RLock()


def normalize_text(value: Any) -> str:
    """Normalize text for Vietnamese lexical matching."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().strip().split())


def tokenize(text: str) -> List[str]:
    """Tokenize Vietnamese text using a lightweight Unicode regex."""
    text = normalize_text(text)
    return [
        token
        for token in re.findall(r"[\wÀ-ỹ]+", text, flags=re.UNICODE)
        if len(token) >= 2
    ]


def normalize_list(values: Optional[List[str]]) -> List[str]:
    """Normalize and de-duplicate a list of strings."""
    if not values:
        return []
    results: List[str] = []
    for item in values:
        text = normalize_text(item)
        if text and text not in results:
            results.append(text)
    return results


def load_knowledge_base(force_reload: bool = False) -> List[Dict[str, Any]]:
    """Load the JSONL knowledge base once and cache it in memory."""
    global _knowledge_cache
    with _cache_lock:
        if _knowledge_cache is not None and not force_reload:
            return _knowledge_cache

        if not KNOWLEDGE_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy file tri thức tại: {KNOWLEDGE_PATH}")

        docs: List[Dict[str, Any]] = []
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as file_obj:
            for idx, line in enumerate(file_obj, start=1):
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                doc.setdefault("evidence_id", f"KB{idx}")
                docs.append(doc)

        # Nạp thêm bộ giải thích đủ 49 bệnh DDXPlus nếu file tồn tại.
        # Cơ chế này tránh trường hợp bệnh Top-1/Top-3 không có context RAG.
        existing_ids = {str(doc.get("evidence_id")) for doc in docs if doc.get("evidence_id")}
        if DDXPLUS_DISEASE_SUPPLEMENT_PATH.exists():
            appended = 0
            with open(DDXPLUS_DISEASE_SUPPLEMENT_PATH, "r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    evidence_id = str(doc.get("evidence_id") or "")
                    if evidence_id and evidence_id in existing_ids:
                        continue
                    if evidence_id:
                        existing_ids.add(evidence_id)
                    else:
                        doc.setdefault("evidence_id", f"DDXPLUS_DISEASE_AUTO_{len(existing_ids) + 1}")
                    docs.append(doc)
                    appended += 1
            if appended:
                logger.info(
                    "Appended %s DDXPlus disease RAG supplement documents from %s",
                    appended,
                    DDXPLUS_DISEASE_SUPPLEMENT_PATH,
                )
        else:
            logger.warning(
                "DDXPlus disease RAG supplement file not found at %s",
                DDXPLUS_DISEASE_SUPPLEMENT_PATH,
            )

        _knowledge_cache = docs
        logger.info("Loaded %s RAG knowledge documents from %s", len(docs), KNOWLEDGE_PATH)
        return _knowledge_cache


def load_disease_aliases(force_reload: bool = False) -> Dict[str, List[str]]:
    """Load disease aliases used to expand retriever disease queries."""
    global _alias_cache
    with _cache_lock:
        if _alias_cache is not None and not force_reload:
            return _alias_cache

        if not ALIAS_PATH.exists():
            logger.warning("Disease alias mapping not found at %s", ALIAS_PATH)
            _alias_cache = {}
        else:
            with open(ALIAS_PATH, "r", encoding="utf-8") as file_obj:
                _alias_cache = json.load(file_obj)
        return _alias_cache


def expand_disease_queries(diseases: List[str]) -> List[str]:
    aliases = load_disease_aliases()
    expanded: List[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in expanded:
            expanded.append(text)

    for disease in diseases or []:
        disease_text = str(disease or "").strip()
        disease_norm = normalize_text(disease_text)
        add(disease_text)

        # Alias từ file cũ nếu vẫn còn tồn tại.
        for alias in aliases.get(disease_text, []) or aliases.get(disease_norm, []) or []:
            add(alias)

        # Alias Việt-Anh tích hợp trong code để không phụ thuộc file keyword cũ.
        for key, values in DEFAULT_DISEASE_TRANSLATION_ALIASES.items():
            key_norm = normalize_text(key)
            value_norms = [normalize_text(item) for item in values]
            if disease_norm == key_norm or disease_norm in value_norms or key_norm in disease_norm:
                add(key)
                for alias in values:
                    add(alias)

    return expanded


def _doc_search_text(doc: Dict[str, Any]) -> str:
    """Build the lexical search text for one knowledge document."""
    fields = [
        doc.get("title", ""),
        doc.get("content", ""),
        doc.get("department", ""),
        " ".join(doc.get("symptoms", []) or []),
        " ".join(doc.get("red_flags", []) or []),
        " ".join(doc.get("disease_names", []) or []),
        " ".join(doc.get("disease_aliases", []) or []),
    ]
    return " ".join(str(item) for item in fields if item)


def build_hybrid_index(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build BM25-lite and TF-IDF structures for cached retrieval."""
    doc_vectors: List[Counter] = []
    doc_lengths: List[int] = []
    df: Counter = Counter()

    for doc in docs:
        tokens = tokenize(_doc_search_text(doc))
        counts = Counter(tokens)
        doc_vectors.append(counts)
        doc_lengths.append(sum(counts.values()))
        for token in counts.keys():
            df[token] += 1

    total_docs = max(len(docs), 1)
    idf = {
        token: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
        for token, freq in df.items()
    }
    avg_doc_len = sum(doc_lengths) / max(len(doc_lengths), 1)
    return {
        "vectors": doc_vectors,
        "lengths": doc_lengths,
        "idf": idf,
        "avg_doc_len": avg_doc_len,
        "document_count": len(docs),
        "vocabulary_size": len(idf),
    }


def initialize_retriever_cache(force_reload: bool = False) -> Dict[str, Any]:
    """Warm the RAG retriever cache during FastAPI startup."""
    global _index_cache, _vector_payload_cache
    with _cache_lock:
        docs = load_knowledge_base(force_reload=force_reload)
        load_disease_aliases(force_reload=force_reload)
        if _index_cache is None or force_reload:
            _index_cache = build_hybrid_index(docs)
            logger.info(
                "Built RAG lexical index: documents=%s, vocabulary=%s",
                _index_cache["document_count"],
                _index_cache["vocabulary_size"],
            )
        if _vector_payload_cache is None or force_reload:
            _vector_payload_cache = prepare_vector_documents(docs)
        return {
            "documents": len(docs),
            "vocabulary_size": _index_cache.get("vocabulary_size", 0),
            "vector_ready_documents": len(_vector_payload_cache),
        }


def get_hybrid_index() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return cached knowledge docs and lexical index."""
    global _index_cache
    docs = load_knowledge_base()
    if _index_cache is None:
        initialize_retriever_cache()
    if _index_cache is None:
        raise RuntimeError("RAG index was not initialized.")
    return docs, _index_cache


def prepare_vector_documents(
    docs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    source_docs = docs if docs is not None else load_knowledge_base()
    vector_docs: List[Dict[str, Any]] = []
    for idx, doc in enumerate(source_docs, start=1):
        evidence_id = str(doc.get("evidence_id") or f"KB{idx}")
        title = str(doc.get("title") or "")
        content = str(doc.get("content") or "")
        text = "\n".join(part for part in [title, content] if part).strip()
        if not text:
            text = _doc_search_text(doc)
        vector_docs.append(
            {
                "id": evidence_id,
                "text": text,
                "metadata": {
                    "title": title,
                    "source": doc.get("source"),
                    "department": doc.get("department"),
                    "symptoms": doc.get("symptoms", []) or [],
                    "red_flags": doc.get("red_flags", []) or [],
                    "disease_names": doc.get("disease_names", []) or [],
                },
            }
        )
    return vector_docs


def get_vector_store_payload() -> List[Dict[str, Any]]:
    """Return cached VectorDB-ready records."""
    global _vector_payload_cache
    if _vector_payload_cache is None:
        initialize_retriever_cache()
    return _vector_payload_cache or []


def build_vector_db_payload(output_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Create optional JSONL payload for a future vector-store build job."""
    payload = get_vector_store_payload()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file_obj:
            for item in payload:
                file_obj.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Wrote VectorDB preparation payload to %s", output_path)
    return payload


def get_retriever_status() -> Dict[str, Any]:
    """Return operational status for diagnostics and health checks."""
    if _index_cache is None:
        return {"initialized": False, "knowledge_path": str(KNOWLEDGE_PATH)}
    return {
        "initialized": True,
        "knowledge_path": str(KNOWLEDGE_PATH),
        "documents": _index_cache.get("document_count", 0),
        "vocabulary_size": _index_cache.get("vocabulary_size", 0),
        "vector_ready_documents": len(get_vector_store_payload()),
        "vector_backend": "not_configured",
    }


def bm25_score(
    query_tokens: List[str],
    doc_counter: Counter,
    doc_len: int,
    avg_doc_len: float,
    idf: Dict[str, float],
) -> float:
    """Compute a lightweight BM25 score against one cached document vector."""
    if not query_tokens:
        return 0.0
    k1 = 1.5
    b = 0.75
    score = 0.0
    query_counts = Counter(query_tokens)
    for token, qf in query_counts.items():
        tf = doc_counter.get(token, 0)
        if tf <= 0:
            continue
        token_idf = idf.get(token, 0.0)
        denominator = tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1.0))
        score += token_idf * ((tf * (k1 + 1)) / max(denominator, 1e-9)) * min(qf, 3)
    return score


def tfidf_cosine(query_tokens: List[str], doc_counter: Counter, idf: Dict[str, float]) -> float:
    """Compute TF-IDF cosine similarity against one cached document vector."""
    if not query_tokens or not doc_counter:
        return 0.0
    query_counts = Counter(query_tokens)

    numerator = 0.0
    query_norm = 0.0
    doc_norm = 0.0

    for token, q_count in query_counts.items():
        q_weight = q_count * idf.get(token, 0.0)
        d_weight = doc_counter.get(token, 0) * idf.get(token, 0.0)
        numerator += q_weight * d_weight
        query_norm += q_weight * q_weight

    for token, d_count in doc_counter.items():
        weight = d_count * idf.get(token, 0.0)
        doc_norm += weight * weight

    if query_norm <= 0 or doc_norm <= 0:
        return 0.0
    return numerator / (math.sqrt(query_norm) * math.sqrt(doc_norm))


def structured_score(
    doc: Dict[str, Any],
    symptoms: List[str],
    red_flags: List[str],
    department: Optional[str],
    diseases: List[str],
) -> float:
    """Score exact structured matches before lexical scores are added."""
    score = 0.0

    doc_symptoms = normalize_list(doc.get("symptoms", []))
    doc_red_flags = normalize_list(doc.get("red_flags", []))
    doc_disease_names = normalize_list(doc.get("disease_names", []))
    doc_disease_aliases = normalize_list(doc.get("disease_aliases", []))
    doc_department = normalize_text(doc.get("department"))
    doc_source = normalize_text(doc.get("source"))

    symptom_set = set(normalize_list(symptoms))
    red_flag_set = set(normalize_list(red_flags))
    disease_set = set(normalize_list(diseases))
    department_text = normalize_text(department)

    title = normalize_text(doc.get("title"))
    content = normalize_text(doc.get("content"))

    symptom_matches = symptom_set.intersection(doc_symptoms)
    red_flag_matches = red_flag_set.intersection(doc_red_flags)
    disease_name_matches = disease_set.intersection(doc_disease_names)
    disease_alias_matches = disease_set.intersection(doc_disease_aliases)

    score += len(symptom_matches) * 6
    score += len(red_flag_matches) * 8
    score += len(disease_name_matches) * 12
    score += len(disease_alias_matches) * 9

    if department_text and doc_department == department_text:
        score += 5

    disease_hit = bool(disease_name_matches or disease_alias_matches)
    for disease in disease_set:
        if disease and disease in title:
            score += 10
            disease_hit = True
        elif disease and disease in content:
            score += 4
            disease_hit = True

    for symptom in symptom_set:
        if symptom and symptom in title:
            score += 3
        if symptom and symptom in content:
            score += 2

    for flag in red_flag_set:
        if flag and flag in content:
            score += 2

    if doc_source == "knowledge_base_internal":
        score += 4
    if doc_source == "medlineplus" and disease_set and not disease_hit:
        score -= 2
    return score




def disease_hit_score(doc: Dict[str, Any], expanded_diseases: List[str]) -> float:
    """Prioritize documents that explicitly mention predicted diseases/aliases.

    RAG++ phải lấy đúng nội dung của bệnh đã dự đoán trước. Vì vậy, khi danh
    sách bệnh đã có, tài liệu khớp tên bệnh/alias trong metadata, tiêu đề hoặc
    nội dung được đưa lên trước tài liệu chỉ khớp triệu chứng chung.
    """
    if not expanded_diseases:
        return 0.0

    doc_disease_names = set(normalize_list(doc.get("disease_names", [])))
    doc_disease_aliases = set(normalize_list(doc.get("disease_aliases", [])))
    title = normalize_text(doc.get("title"))
    content = normalize_text(doc.get("content"))
    disease_terms = normalize_list(expanded_diseases)

    score = 0.0
    for disease in disease_terms:
        disease_norm = normalize_text(disease)
        if not disease_norm:
            continue
        if disease_norm in doc_disease_names:
            score = max(score, 60.0)
        elif disease_norm in doc_disease_aliases:
            score = max(score, 55.0)
        elif disease_norm in title:
            score = max(score, 45.0)
        elif disease_norm in content:
            score = max(score, 25.0)
    return score

def build_query_text(
    symptoms: List[str],
    red_flags: List[str],
    department: Optional[str],
    diseases: List[str],
) -> str:
    """Build query text from structured clinical context."""
    parts: List[str] = []
    parts.extend(symptoms or [])
    parts.extend(red_flags or [])
    parts.extend(diseases or [])
    if department:
        parts.append(department)
    return " ".join(str(item) for item in parts if item)


def retrieve_relevant_documents(
    symptoms: List[str],
    red_flags: Optional[List[str]] = None,
    department: Optional[str] = None,
    diseases: Optional[List[str]] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Retrieve top-k evidence documents from cached RAG structures."""
    docs, index = get_hybrid_index()
    expanded_diseases = expand_disease_queries(diseases or [])
    query_text = build_query_text(symptoms or [], red_flags or [], department, expanded_diseases)
    query_tokens = tokenize(query_text)

    scored_docs: List[Tuple[float, Dict[str, Any]]] = []
    for idx, doc in enumerate(docs):
        doc_counter = index["vectors"][idx]
        doc_len = index["lengths"][idx]
        bm25 = bm25_score(query_tokens, doc_counter, doc_len, index["avg_doc_len"], index["idf"])
        cosine = tfidf_cosine(query_tokens, doc_counter, index["idf"])
        structured = structured_score(
            doc=doc,
            symptoms=symptoms or [],
            red_flags=red_flags or [],
            department=department,
            diseases=expanded_diseases,
        )
        disease_boost = disease_hit_score(doc, expanded_diseases)

        # Nếu có bệnh dự đoán, ưu tiên tài liệu khớp bệnh; nếu không có bệnh
        # thì dùng truy xuất theo triệu chứng như cũ.
        total = disease_boost + structured + bm25 * 1.5 + cosine * 8
        if total > 0:
            doc_copy = dict(doc)
            doc_copy["retrieval_score"] = round(total, 4)
            doc_copy["retrieval_features"] = {
                "disease_boost": round(disease_boost, 4),
                "structured": round(structured, 4),
                "bm25": round(bm25, 4),
                "tfidf_cosine": round(cosine, 4),
            }
            scored_docs.append((total, doc_copy))

    # Nhóm tài liệu khớp bệnh trước, rồi mới đến tài liệu chỉ khớp triệu chứng.
    disease_scored = [item for item in scored_docs if item[1].get("retrieval_features", {}).get("disease_boost", 0) > 0]
    symptom_scored = [item for item in scored_docs if item not in disease_scored]
    disease_scored.sort(key=lambda item: item[0], reverse=True)
    symptom_scored.sort(key=lambda item: item[0], reverse=True)
    ordered_docs = disease_scored + symptom_scored if expanded_diseases else sorted(scored_docs, key=lambda item: item[0], reverse=True)

    limit = max(1, min(int(top_k or 3), 8))
    return [doc for _, doc in ordered_docs[:limit]]
