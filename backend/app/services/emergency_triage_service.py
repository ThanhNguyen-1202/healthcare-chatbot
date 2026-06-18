from typing import Any, Dict

from app.services.triage_service import evaluate_triage


def check_emergency_triage(collected_data: Dict[str, Any]) -> Dict[str, Any]:
    triage = evaluate_triage(collected_data, base_level="Tự theo dõi")

    if triage.get("triage_priority") == 1:
        return {
            "is_emergency": True,
            "matched_flags": triage.get("reasons", []),
            "triage_level": triage.get("triage_level", "Cấp cứu ngay"),
            "triage_priority": triage.get("triage_priority", 1),
            "triage_label": triage.get("triage_label", "Mức 1 - Cấp cứu ngay"),
            "department": "Cấp cứu",
            "warning": (
                "Có dấu hiệu nguy hiểm nghiêm trọng. "
                "Người bệnh nên đến cơ sở y tế hoặc gọi 115 ngay."
            ),
            "action_recommendation": triage.get("action_recommendation"),
            "next_step": "emergency_override",
        }

    return {
        "is_emergency": False,
        "matched_flags": [],
        "triage_level": triage.get("triage_level"),
        "triage_priority": triage.get("triage_priority"),
        "triage_label": triage.get("triage_label"),
        "department": None,
        "warning": None,
        "action_recommendation": triage.get("action_recommendation"),
        "next_step": "continue",
    }
