from typing import Any, Dict


MEDICAL_DISCLAIMER = (
    "Cảnh báo y tế: Kết quả chỉ hỗ trợ sàng lọc và tham khảo ban đầu, "
    "không thay thế chẩn đoán hoặc điều trị của bác sĩ. Nếu có dấu hiệu "
    "nguy hiểm như khó thở nặng, đau ngực dữ dội, lơ mơ, ngất, co giật, "
    "chảy máu nhiều hoặc triệu chứng diễn tiến nhanh, hãy gọi 115 hoặc "
    "đến cơ sở cấp cứu ngay."
)


def with_medical_disclaimer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the standard medical disclaimer to a response dictionary."""
    result = dict(payload)
    result.setdefault("medical_disclaimer", MEDICAL_DISCLAIMER)
    result.setdefault("safety_notice", MEDICAL_DISCLAIMER)
    return result
