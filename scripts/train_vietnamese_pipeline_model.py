"""Train the Vietnamese DDXPlus text classifier with a held-out test split.

This model is an auxiliary component for the natural-language pipeline.  It is
not a replacement for the structured DDXPlus model and must not be reported as
clinical accuracy on real patients.

Run from project root:
    python scripts/train_vietnamese_pipeline_model.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "DDXPlus_Benchmark_1000_Cau_Giong_File_Mau.xlsx"
DEFAULT_MAPPING = PROJECT_ROOT / "backend/app/mappings/ddxplus_disease_mapping.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "backend/app/ml/artifacts/ddxplus_vn_text_model.pkl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_excel(args.input, sheet_name="scoring_ready")
    required = {"question", "ground_truth"}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Thiếu cột: {sorted(missing_columns)}")
    data = data.dropna(subset=["question", "ground_truth"]).copy()
    data["question"] = data["question"].astype(str)
    data["ground_truth"] = data["ground_truth"].astype(str)

    mapping = json.loads(DEFAULT_MAPPING.read_text(encoding="utf-8"))
    vi_to_canonical = {
        str(item.get("vi") or "").strip(): canonical
        for canonical, item in mapping.items()
        if str(item.get("vi") or "").strip()
    }
    data["canonical"] = data["ground_truth"].map(vi_to_canonical)
    if data["canonical"].isna().any():
        names = sorted(data.loc[data["canonical"].isna(), "ground_truth"].unique())
        raise ValueError("Thiếu mapping bệnh: " + ", ".join(names))

    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=data["canonical"],
    )

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, max_df=0.98,
                sublinear_tf=True, max_features=45_000,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                sublinear_tf=True, max_features=80_000,
            )),
        ])),
        ("model", LogisticRegression(
            max_iter=3000, C=6.0, class_weight="balanced",
            random_state=args.random_state,
        )),
    ])

    pipeline.fit(data.loc[train_idx, "question"], data.loc[train_idx, "canonical"])
    probabilities = pipeline.predict_proba(data.loc[test_idx, "question"])
    predictions = pipeline.classes_[np.argmax(probabilities, axis=1)]
    y_true = data.loc[test_idx, "canonical"].to_numpy()
    class_to_index = {name: idx for idx, name in enumerate(pipeline.classes_)}
    y_indices = np.asarray([class_to_index[name] for name in y_true])

    metrics = {
        "dataset_rows": int(len(data)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "classes": int(len(pipeline.classes_)),
        "random_state": args.random_state,
        "test_size": args.test_size,
        "top1_accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "top3_accuracy": float(top_k_accuracy_score(
            y_indices, probabilities, k=3, labels=np.arange(len(pipeline.classes_))
        )),
    }

    artifact = {
        "pipeline": pipeline,
        "classes": list(pipeline.classes_),
        "disease_mapping": mapping,
        "metrics": metrics,
        "train_indices": [int(value) for value in train_idx],
        "test_indices": [int(value) for value in test_idx],
        "source_file": args.input.name,
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "label_type": "DDXPlus canonical disease",
        "scope": "Vietnamese synthetic benchmark text classifier; held-out evaluation",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    metrics_path = args.output.with_name("ddxplus_vn_text_model_metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = args.output.with_name("ddxplus_vn_text_model_report.txt")
    report_path.write_text(
        classification_report(y_true, predictions, labels=pipeline.classes_, zero_division=0),
        encoding="utf-8",
    )
    split_path = PROJECT_ROOT / "data/processed/ddxplus_vn_pipeline_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({
        "train_indices_zero_based": artifact["train_indices"],
        "test_indices_zero_based": artifact["test_indices"],
        "metrics": metrics,
        "source_file": artifact["source_file"],
        "source_sha256": artifact["source_sha256"],
        "note": "Test indices were not used to fit the Vietnamese text model.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved model: {args.output}")
    print(f"Saved split: {split_path}")


if __name__ == "__main__":
    main()
