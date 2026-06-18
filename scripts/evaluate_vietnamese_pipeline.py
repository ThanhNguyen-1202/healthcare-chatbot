"""Evaluate only the held-out Vietnamese test rows.

Local mode evaluates the deterministic text/hybrid service without calling
Gemini. API mode calls /predict/ragpp with use_llm_extractor=false so repeated
runs are reproducible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.vn_disease_text_service import predict_vietnamese_text_diseases

DEFAULT_INPUT = PROJECT_ROOT / "DDXPlus_Benchmark_1000_Cau_Giong_File_Mau.xlsx"
DEFAULT_MODEL = PROJECT_ROOT / "backend/app/ml/artifacts/ddxplus_vn_text_model.pkl"


def evaluate_local(data: pd.DataFrame, test_indices: list[int]) -> dict:
    artifact = joblib.load(DEFAULT_MODEL)
    mapping = artifact["disease_mapping"]
    vi_to_canonical = {
        str(item.get("vi") or "").strip(): canonical
        for canonical, item in mapping.items()
    }
    y_true = []
    y_pred = []
    ranked_names = []
    for index in test_indices:
        row = data.iloc[index]
        predictions = predict_vietnamese_text_diseases(str(row["question"]), top_k=3)
        y_true.append(vi_to_canonical[str(row["ground_truth"]).strip()])
        y_pred.append(predictions[0]["canonical_name"])
        ranked_names.append([item["canonical_name"] for item in predictions])

    classes = sorted(set(y_true) | {name for row in ranked_names for name in row})
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    proba = np.zeros((len(y_true), len(classes)), dtype=float)
    for row_index, names in enumerate(ranked_names):
        for rank, name in enumerate(names):
            proba[row_index, class_to_idx[name]] = 1.0 / (rank + 1)
    true_idx = np.asarray([class_to_idx[name] for name in y_true])
    return {
        "rows": len(y_true),
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "top3_hit_rate": float(top_k_accuracy_score(
            true_idx, proba, k=3, labels=np.arange(len(classes))
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    data = pd.read_excel(args.input, sheet_name="scoring_ready")
    artifact = joblib.load(DEFAULT_MODEL)
    metrics = evaluate_local(data, artifact["test_indices"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Đây là kết quả trên 20% dữ liệu giữ lại, không dùng để huấn luyện.")


if __name__ == "__main__":
    main()
