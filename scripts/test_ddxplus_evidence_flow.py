"""Smoke test for Vietnamese text -> DDXPlus evidence -> Top-3 prediction.

Run from the project root after activating the backend virtual environment:
    python scripts/test_ddxplus_evidence_flow.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.llm_extractor_service import extract_intake_with_llm
from app.services.ml_disease_prediction_service import predict_top_diseases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="Tôi là nam 30 tuổi, bị sốt 39 độ, ho có đờm và khó thở.",
    )
    args = parser.parse_args()

    intake = extract_intake_with_llm(args.text)
    print("INTAKE:")
    print(json.dumps(intake, ensure_ascii=False, indent=2))

    evidence_codes = intake.get("ddxplus_evidences", []) or []
    if not evidence_codes:
        print("\nFAILED: Không trích xuất được ddxplus_evidences.")
        return 2

    ranked = predict_top_diseases(
        text=args.text,
        symptoms=[intake.get("main_symptom") or args.text],
        top_k=3,
        age=intake.get("age"),
        gender=intake.get("gender"),
        evidence_codes=evidence_codes,
    )
    print("\nTOP-3:")
    print(json.dumps(ranked, ensure_ascii=False, indent=2))
    print("\nSUCCESS: Luồng evidence và mô hình DDXPlus hoạt động.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
