"""Prediction API routes."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from pydantic import Field

from app.core.api_response import api_response
from app.core.config import settings
from app.core.medical_disclaimer import MEDICAL_DISCLAIMER
from app.core.rate_limit import limiter
from app.db.repositories.prediction_repo import PredictionRepository
from app.rag.explainer import build_rag_explanation
from app.rag.retriever import retrieve_relevant_documents
from app.schemas.prediction import PredictionRequest
from app.services.intake_service import build_intake_from_messages
from app.services.ml_disease_prediction_service import predict_top_diseases
from app.services.hybrid_disease_prediction_service import predict_hybrid_diseases
from app.services.ddxplus_evidence_catalog import normalize_evidence_codes
from app.services.prediction_service import predict_from_symptoms

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["prediction"])


class RAGPPPredictionRequest(PredictionRequest):
    """Prediction input that also accepts a free-text clinical message."""

    message: Optional[str] = Field(
        default=None,
        description="Mô tả triệu chứng tự nhiên. Có thể dùng thay cho danh sách symptoms.",
    )
    prediction_mode: str = Field(
        default="hybrid",
        description="hybrid | text | structured",
    )
    use_llm_extractor: bool = Field(
        default=True,
        description="Tắt khi benchmark để chạy xác định, không phụ thuộc Gemini.",
    )


def _collect_symptoms_from_intake(intake: Dict[str, Any]) -> List[str]:
    """Collect main and secondary symptoms from intake output."""
    symptoms: List[str] = []
    main_symptom = intake.get("main_symptom")
    if main_symptom:
        symptoms.append(str(main_symptom))
    for symptom in intake.get("secondary_symptoms", []) or []:
        if symptom and symptom not in symptoms:
            symptoms.append(str(symptom))
    return symptoms


def _normalize_top_diseases(items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Normalize disease ranking items for frontend and reporting."""
    limit = max(1, min(int(top_k or 3), 5))
    results: List[Dict[str, Any]] = []
    for item in items[:limit]:
        name = item.get("name") or item.get("label")
        if not name:
            continue
        score = item.get("score", 0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        results.append(
            {
                "name": str(name),
                "canonical_name": item.get("canonical_name"),
                "score": round(score, 6),
                "confidence": round(float(item.get("confidence", score)), 6),
                "percent": round(float(item.get("percent", score * 100)), 2),
                "icd10": item.get("icd10"),
                "severity": item.get("severity"),
                "department": item.get("department"),
                "source": item.get("source", "rule_or_ml"),
            }
        )
    return results


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _required_input_missing(age: Any, gender: Any, symptoms: List[str]) -> List[str]:

    missing: List[str] = []
    if _is_missing_value(age):
        missing.append("age")
    if _is_missing_value(gender):
        missing.append("gender")
    if _is_missing_value(symptoms):
        missing.append("symptoms")
    return missing


def _missing_field_question(missing: List[str]) -> str:
    labels = {
        "age": "tuổi",
        "gender": "giới tính",
        "symptoms": "triệu chứng",
    }
    questions = {
        "age": "Anh/chị vui lòng cho biết tuổi của người đang có triệu chứng.",
        "gender": "Anh/chị vui lòng cho biết giới tính của người đang có triệu chứng: nam hoặc nữ.",
        "symptoms": "Anh/chị vui lòng mô tả triệu chứng chính hiện tại, ví dụ: sốt, ho, đau bụng, đau ngực hoặc khó thở.",
    }
    first = missing[0] if missing else "symptoms"
    return (
        "Thiếu dữ liệu bắt buộc trước khi dự đoán: "
        + ", ".join(labels.get(item, item) for item in missing)
        + ". "
        + questions.get(first, "Anh/chị vui lòng cung cấp thêm thông tin.")
    )


@router.post("/")
@limiter.limit(settings.predict_rate_limit)
async def predict_endpoint(request: Request, payload: PredictionRequest) -> dict:
    """Predict triage and possible diseases with per-client rate limiting."""
    missing_required = _required_input_missing(payload.age, payload.gender, payload.symptoms)
    if missing_required:
        return api_response(
            data={
                "matched": False,
                "missing_fields": missing_required,
                "next_step": "need_required_fields",
                "message": _missing_field_question(missing_required),
            },
            message="Missing mandatory clinical fields",
        )

    result = predict_from_symptoms(
        symptoms=payload.symptoms,
        red_flags=payload.red_flags,
        temperature=payload.temperature,
        pain_score=payload.pain_score,
        duration=payload.duration,
        comorbidities=payload.comorbidities,
        medications=payload.medications,
        age=payload.age,
        gender=payload.gender,
        is_pregnant=payload.is_pregnant,
        top_k=payload.top_k,
    )
    if payload.evidence_codes:
        ranked = predict_top_diseases(
            symptoms=payload.symptoms,
            top_k=payload.top_k,
            age=payload.age,
            gender=payload.gender,
            evidence_codes=payload.evidence_codes,
        )
        result["possible_diseases"] = _normalize_top_diseases(ranked, payload.top_k)
        result["matched"] = bool(result["possible_diseases"])
        result["message"] = "Đã dự đoán bằng mô hình DDXPlus có cấu trúc."
        result["model_source"] = "ddxplus_structured_sgd"
        result["ddxplus_evidences"] = payload.evidence_codes
    return api_response(data=result, message="Prediction generated")


@router.post("/ragpp")
@limiter.limit(settings.predict_rate_limit)
async def predict_ragpp_endpoint(request: Request, payload: RAGPPPredictionRequest) -> dict:
    """Run RAG++ prediction flow with per-client rate limiting."""
    intake: Dict[str, Any] = {}
    if payload.message:
        intake = build_intake_from_messages(
            [{"role": "user", "content": payload.message}],
            use_llm=payload.use_llm_extractor,
        )

    symptoms = payload.symptoms or _collect_symptoms_from_intake(intake)
    red_flags = payload.red_flags or intake.get("red_flags", []) or []
    temperature = payload.temperature if payload.temperature is not None else intake.get("temperature")
    pain_score = payload.pain_score if payload.pain_score is not None else intake.get("pain_score")  # [CHANGED] Giữ điểm đau 0-10.
    duration = payload.duration or intake.get("duration")
    comorbidities = payload.comorbidities or intake.get("comorbidities", []) or []
    medications = payload.medications or intake.get("medications", []) or []
    age = payload.age if payload.age is not None else intake.get("age")
    gender = payload.gender or intake.get("gender")
    is_pregnant = payload.is_pregnant if payload.is_pregnant is not None else intake.get("is_pregnant")
    evidence_codes = normalize_evidence_codes(
        payload.evidence_codes or intake.get("ddxplus_evidences", []) or []
    )
    top_k = max(1, min(int(payload.top_k or 5), 5))

    missing_required = _required_input_missing(age, gender, symptoms)
    if missing_required:
        return api_response(
            data={
                "input_mode": "message" if payload.message else "structured",
                "intake": intake,
                "symptoms": symptoms,
                "ddxplus_evidences": evidence_codes,
                "missing_fields": missing_required,
                "next_step": "need_required_fields",
                "question": _missing_field_question(missing_required),
                "top_diseases": [],
                "prediction": {},
                "ragpp": {
                    "retrieval": "skipped",
                    "evidence": [],
                    "explanation": "Chưa chạy RAG++ vì còn thiếu tuổi, giới tính hoặc triệu chứng.",
                },
                "safety_notice": MEDICAL_DISCLAIMER,
                "medical_disclaimer": MEDICAL_DISCLAIMER,
            },
            message="Missing mandatory clinical fields",
            metadata={"retrieval_backend": "skipped"},
        )

    prediction = predict_from_symptoms(
        symptoms=symptoms,
        red_flags=red_flags,
        temperature=temperature,
        pain_score=pain_score,  # [CHANGED] Truyền điểm đau 0-10 vào pipeline.
        duration=duration,
        comorbidities=comorbidities,
        medications=medications,
        age=age,
        gender=gender,
        is_pregnant=is_pregnant,
        top_k=top_k,
    )

    ml_text = payload.message or " ".join(symptoms + red_flags + comorbidities)
    try:
        ranked_diseases = predict_hybrid_diseases(
            ml_text,
            symptoms=symptoms,
            top_k=top_k,
            age=age,
            gender=gender,
            evidence_codes=evidence_codes,
            prediction_mode=payload.prediction_mode,
        )
        if not ranked_diseases:
            ranked_diseases = prediction.get("possible_diseases", []) or []
    except Exception as exc:
        logger.exception("Hybrid disease prediction failed in RAG++ endpoint")
        ranked_diseases = prediction.get("possible_diseases", []) or []
        prediction.setdefault("warnings", []).append(str(exc))

    top_diseases = _normalize_top_diseases(ranked_diseases, top_k=top_k)
    rag_query_disease_names: List[str] = []
    display_disease_names: List[str] = []
    for item in top_diseases:
        display_name = item.get("name") or item.get("canonical_name")
        if display_name and display_name not in display_disease_names:
            display_disease_names.append(display_name)
        for key in ["name", "canonical_name"]:
            value = item.get(key)
            if value and value not in rag_query_disease_names:
                rag_query_disease_names.append(value)
    docs = retrieve_relevant_documents(
        symptoms=symptoms,
        red_flags=red_flags,
        department=prediction.get("department"),
        diseases=rag_query_disease_names,
        top_k=8,
    )
    explanation = build_rag_explanation(
        docs,
        disease_names=display_disease_names,
        symptoms=symptoms,
        red_flags=red_flags,
        department=prediction.get("department"),
    )

    return api_response(
        data={
            "input_mode": "message" if payload.message else "structured",
            "intake": intake,
            "symptoms": symptoms,
            "ddxplus_evidences": evidence_codes,
            "ddxplus_negative_evidences": intake.get("ddxplus_negative_evidences", []),
            "prediction_mode": payload.prediction_mode,
            "text_model_used": any(
                item.get("source") in {"vn_text_tfidf_lr", "hybrid_vn_text_ddxplus"}
                for item in top_diseases
            ),
            "top_diseases": top_diseases,
            "prediction": prediction,
            "ragpp": {
                "retrieval": "startup-cached hybrid BM25-lite + TF-IDF + structured rerank",
                "evidence": docs,
                "explanation": explanation,
            },
            "safety_notice": MEDICAL_DISCLAIMER,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        },
        message="RAG++ prediction generated",
        metadata={"retrieval_backend": "cached_lexical_hybrid"},
    )


@router.get("/session/{session_id}")
async def get_predictions_by_session(session_id: str) -> dict:
    """Return prediction documents associated with one session."""
    prediction_repo = PredictionRepository()
    predictions = await prediction_repo.get_predictions_by_session(session_id)

    for item in predictions:
        if "_id" in item:
            item["_id"] = str(item["_id"])

    return api_response(
        data={
            "session_id": session_id,
            "total": len(predictions),
            "predictions": predictions,
        },
        message="Session predictions loaded",
    )
