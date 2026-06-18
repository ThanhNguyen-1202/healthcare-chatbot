"""Regression checks for the upgraded Vietnamese pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.services.hybrid_disease_prediction_service import predict_hybrid_diseases
from app.services.vn_evidence_candidate_service import extract_rule_evidences


def main() -> None:
    model_path = PROJECT_ROOT / "backend/app/ml/artifacts/ddxplus_vn_text_model.pkl"
    benchmark_path = PROJECT_ROOT / "DDXPlus_Benchmark_1000_Cau_Giong_File_Mau.xlsx"
    artifact = joblib.load(model_path)
    data = pd.read_excel(benchmark_path, sheet_name="scoring_ready")

    y_true = []
    y_pred = []
    for index in artifact["test_indices"]:
        row = data.iloc[index]
        evidence = extract_rule_evidences(str(row["question"]))["positive"]
        prediction = predict_hybrid_diseases(
            str(row["question"]),
            top_k=1,
            evidence_codes=evidence,
            prediction_mode="hybrid",
        )
        y_true.append(str(row["ground_truth"]))
        y_pred.append(prediction[0]["name"] if prediction else "")

    accuracy = accuracy_score(y_true, y_pred)
    print(json.dumps({"rows": len(y_true), "top1_accuracy": accuracy}, ensure_ascii=False, indent=2))
    if accuracy < 0.90:
        raise SystemExit(f"Regression failed: accuracy={accuracy:.4f} < 0.90")

    negation = extract_rule_evidences(
        "Sưng bàn tay nhưng không khó thở, không ho ra máu và không sốt."
    )
    assert "E_151" in negation["positive"]
    assert "E_66" in negation["negative"]
    assert "E_45" in negation["negative"]
    assert "E_91" in negation["negative"]
    print("Regression checks passed.")


if __name__ == "__main__":
    main()
