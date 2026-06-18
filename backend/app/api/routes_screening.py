from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.medical_disclaimer import MEDICAL_DISCLAIMER, with_medical_disclaimer
from app.rag.explainer import build_rag_explanation
from app.rag.retriever import retrieve_relevant_documents
from app.schemas.prediction import PredictionRequest
from app.services.intake_service import build_intake_from_messages, get_missing_fields
from app.services.ml_disease_prediction_service import predict_top_diseases
from app.services.prediction_service import predict_from_symptoms

router = APIRouter(prefix="/screening", tags=["screening"])


class ScreeningAnalyzeRequest(BaseModel):
    message: str = Field(..., description="Mô tả triệu chứng bằng tiếng Việt tự nhiên")
    top_k: int = Field(default=5, ge=1, le=5)


class ScreeningTriageRequest(PredictionRequest):
    message: Optional[str] = Field(default=None, description="Có thể nhập câu mô tả tự nhiên thay cho symptoms")


def _collect_symptoms(intake: Dict[str, Any]) -> List[str]:
    symptoms: List[str] = []
    main = intake.get("main_symptom")
    if main:
        symptoms.append(str(main))
    for symptom in intake.get("secondary_symptoms", []) or []:
        if symptom and symptom not in symptoms:
            symptoms.append(str(symptom))
    return symptoms


def _run_screening_pipeline(
    *,
    message: Optional[str] = None,
    symptoms: Optional[List[str]] = None,
    red_flags: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    pain_score: Optional[int] = None,  # [CHANGED] Nhận điểm đau 0-10, không nhận severity.
    duration: Optional[str] = None,
    comorbidities: Optional[List[str]] = None,
    medications: Optional[List[str]] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    is_pregnant: Optional[bool] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    intake: Dict[str, Any] = {}
    if message:
        intake = build_intake_from_messages([{"role": "user", "content": message}])

    final_symptoms = symptoms or _collect_symptoms(intake)
    final_red_flags = red_flags or intake.get("red_flags", []) or []
    final_temperature = temperature if temperature is not None else intake.get("temperature")
    final_pain_score = pain_score if pain_score is not None else intake.get("pain_score")  # [CHANGED] Giữ điểm đau 0-10.
    final_duration = duration or intake.get("duration")
    final_comorbidities = comorbidities or intake.get("comorbidities", []) or []
    final_medications = medications or intake.get("medications", []) or []
    final_age = age if age is not None else intake.get("age")
    final_gender = gender or intake.get("gender")
    final_is_pregnant = is_pregnant if is_pregnant is not None else intake.get("is_pregnant")
    limit = max(1, min(int(top_k or 5), 5))

    prediction = predict_from_symptoms(
        symptoms=final_symptoms,
        red_flags=final_red_flags,
        temperature=final_temperature,
        pain_score=final_pain_score,  # [CHANGED] Truyền điểm đau 0-10.
        duration=final_duration,
        comorbidities=final_comorbidities,
        medications=final_medications,
        age=final_age,
        gender=final_gender,
        is_pregnant=final_is_pregnant,
        top_k=limit,
    )

    ml_text = message or " ".join(final_symptoms + final_red_flags + final_comorbidities)
    top_diseases = predict_top_diseases(ml_text, symptoms=final_symptoms, top_k=limit)
    if not top_diseases:
        top_diseases = prediction.get("possible_diseases", []) or []

    disease_names = [item.get("name") for item in top_diseases if item.get("name")]
    evidence = retrieve_relevant_documents(
        symptoms=final_symptoms,
        red_flags=final_red_flags,
        department=prediction.get("department"),
        diseases=disease_names,
        top_k=4,
    )

    return {
        "intake": intake,
        "missing_fields": get_missing_fields(intake) if intake else {"required": [], "optional": []},
        "symptoms": final_symptoms,
        "red_flags": final_red_flags,
        "top_diseases": top_diseases,
        "department": prediction.get("department"),
        "triage": {
            "level": prediction.get("triage_level"),
            "priority": prediction.get("triage_priority"),
            "label": prediction.get("triage_label"),
            "action": prediction.get("action_recommendation"),
            "reasons": prediction.get("triage_reasons", []),
        },
        "prediction": prediction,
        "ragpp": {
            "method": "hybrid BM25-lite + TF-IDF vector + rerank + red-flag check",
            "evidence": evidence,
            "explanation": build_rag_explanation(evidence),
        },
        "safety_notice": MEDICAL_DISCLAIMER,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }


@router.post("/analyze")
async def analyze_screening(request: ScreeningAnalyzeRequest):
    return with_medical_disclaimer(_run_screening_pipeline(message=request.message, top_k=request.top_k))


@router.post("/triage")
async def triage_screening(request: ScreeningTriageRequest):
    result = _run_screening_pipeline(
        message=request.message,
        symptoms=request.symptoms,
        red_flags=request.red_flags,
        temperature=request.temperature,
        pain_score=request.pain_score,  # [CHANGED] Truyền điểm đau 0-10.
        duration=request.duration,
        comorbidities=request.comorbidities,
        medications=request.medications,
        age=request.age,
        gender=request.gender,
        is_pregnant=request.is_pregnant,
        top_k=request.top_k,
    )
    return with_medical_disclaimer({
        "triage": result["triage"],
        "department": result["department"],
        "red_flags": result["red_flags"],
        "safety_notice": result["safety_notice"],
    })
