"""Train the DDXPlus structured disease classifier.

The script streams the 1M+ patient rows, converts DDXPlus evidence codes into a
small sparse feature matrix, and trains a probabilistic 49-class classifier.
It is designed for laptops with limited RAM/VRAM and does not require a GPU.

Usage from the project root:
    python scripts/train_ddxplus_structured_model.py \
        --dataset-dir data/raw/ddxplus/extracted \
        --epochs 1

Accepted patient filenames (extension is optional):
    release_train_patients(.csv)
    release_validate_patients(.csv)
    release_test_patients(.csv)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "ddxplus" / "extracted"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "backend" / "app" / "ml" / "artifacts"
DEFAULT_DISEASE_MAPPING_PATH = PROJECT_ROOT / "backend" / "app" / "mappings" / "ddxplus_disease_mapping.json"

EVIDENCE_PATTERN = re.compile(r"E_\d+(?:_@_(?:V_\d+|[-+]?\d+(?:\.\d+)?))?")
AGE_BINS = (-1, 1, 4, 12, 17, 24, 34, 44, 54, 64, 74, 200)
AGE_BIN_NAMES = (
    "0-1", "2-4", "5-12", "13-17", "18-24", "25-34",
    "35-44", "45-54", "55-64", "65-74", "75+",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_ddxplus")


def find_patient_file(dataset_dir: Path, split: str) -> Path:
    candidates = [
        dataset_dir / f"release_{split}_patients",
        dataset_dir / f"release_{split}_patients.csv",
        dataset_dir / split / f"release_{split}_patients",
        dataset_dir / split / f"release_{split}_patients.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu {split}. Đã kiểm tra: "
        + ", ".join(str(item) for item in candidates)
    )


def base_evidence_code(token: str) -> str:
    return token.split("_@_", 1)[0]


def parse_evidence_tokens(raw: Any) -> list[str]:
    if not isinstance(raw, str):
        return []
    return EVIDENCE_PATTERN.findall(raw)


def age_bin_name(age: Any) -> str:
    try:
        value = int(age)
    except (TypeError, ValueError):
        value = 0
    idx = int(np.digitize([value], AGE_BINS[1:-1], right=True)[0])
    return AGE_BIN_NAMES[min(max(idx, 0), len(AGE_BIN_NAMES) - 1)]


def load_metadata(dataset_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    search_dirs = [dataset_dir, dataset_dir.parent, PROJECT_ROOT / "data" / "raw" / "ddxplus"]
    conditions_path = next((d / "release_conditions.json" for d in search_dirs if (d / "release_conditions.json").exists()), None)
    evidences_path = next((d / "release_evidences.json" for d in search_dirs if (d / "release_evidences.json").exists()), None)
    if conditions_path is None or evidences_path is None:
        raise FileNotFoundError("Thiếu release_conditions.json hoặc release_evidences.json")
    return (
        json.loads(conditions_path.read_text(encoding="utf-8")),
        json.loads(evidences_path.read_text(encoding="utf-8")),
    )


def build_evidence_vectorizer(train_path: Path, chunksize: int) -> CountVectorizer:
    tokens: set[str] = set()
    for chunk in pd.read_csv(train_path, usecols=["EVIDENCES"], chunksize=chunksize):
        for raw in chunk["EVIDENCES"].astype(str):
            tokens.update(parse_evidence_tokens(raw))
    vocabulary = {token: idx for idx, token in enumerate(sorted(tokens))}
    return CountVectorizer(
        vocabulary=vocabulary,
        token_pattern=r"E_\d+(?:_@_(?:V_\d+|[-+]?\d+(?:\.\d+)?))?",
        lowercase=False,
        binary=True,
        dtype=np.float32,
    )


def vectorize_chunk(
    chunk: pd.DataFrame,
    vectorizer: CountVectorizer,
) -> csr_matrix:
    evidence_matrix = vectorizer.transform(chunk["EVIDENCES"].astype(str))
    n_rows = len(chunk)
    ages = np.clip(chunk["AGE"].to_numpy(dtype=np.float32), 0, 110)
    sex_values = chunk["SEX"].astype(str).str.upper().to_numpy()

    extras = np.zeros((n_rows, len(AGE_BIN_NAMES) + 4), dtype=np.float32)
    for idx, age in enumerate(ages):
        extras[idx, AGE_BIN_NAMES.index(age_bin_name(age))] = 1.0
    extras[:, len(AGE_BIN_NAMES)] = (sex_values == "M").astype(np.float32)
    extras[:, len(AGE_BIN_NAMES) + 1] = (sex_values == "F").astype(np.float32)
    extras[:, len(AGE_BIN_NAMES) + 2] = ages / 110.0
    extras[:, len(AGE_BIN_NAMES) + 3] = np.asarray(evidence_matrix.sum(axis=1)).ravel() / 60.0
    return hstack([evidence_matrix, csr_matrix(extras)], format="csr", dtype=np.float32)


def count_labels(train_path: Path, chunksize: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for chunk in pd.read_csv(train_path, usecols=["PATHOLOGY"], chunksize=chunksize):
        counts.update(chunk["PATHOLOGY"].astype(str))
    return counts


def train_streaming(
    train_path: Path,
    classifier: SGDClassifier,
    encoder: LabelEncoder,
    vectorizer: CountVectorizer,
    class_counts: Counter[str],
    epochs: int,
    chunksize: int,
    random_state: int,
) -> None:
    total = sum(class_counts.values())
    n_classes = len(encoder.classes_)
    # Square-root balancing protects rare diseases without destabilising SGD.
    class_weights = {
        label: min(15.0, max(0.5, math.sqrt(total / (n_classes * count))))
        for label, count in class_counts.items()
    }
    classes = np.arange(n_classes, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    first_batch = True

    for epoch in range(1, epochs + 1):
        started = time.time()
        seen = 0
        for chunk in pd.read_csv(
            train_path,
            usecols=["AGE", "SEX", "PATHOLOGY", "EVIDENCES"],
            chunksize=chunksize,
        ):
            x = vectorize_chunk(chunk, vectorizer)
            y_labels = chunk["PATHOLOGY"].astype(str).to_numpy()
            y = encoder.transform(y_labels)
            weights = np.asarray([class_weights[label] for label in y_labels], dtype=np.float32)
            order = rng.permutation(len(chunk))
            kwargs = {"classes": classes} if first_batch else {}
            classifier.partial_fit(x[order], y[order], sample_weight=weights[order], **kwargs)
            first_batch = False
            seen += len(chunk)
        logger.info("Epoch %s/%s completed: rows=%s, seconds=%.1f", epoch, epochs, seen, time.time() - started)


def evaluate(
    path: Path,
    classifier: SGDClassifier,
    encoder: LabelEncoder,
    vectorizer: CountVectorizer,
    chunksize: int,
) -> dict[str, Any]:
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    proba_parts: list[np.ndarray] = []

    for chunk in pd.read_csv(
        path,
        usecols=["AGE", "SEX", "PATHOLOGY", "EVIDENCES"],
        chunksize=chunksize,
    ):
        x = vectorize_chunk(chunk, vectorizer)
        y = encoder.transform(chunk["PATHOLOGY"].astype(str))
        proba = classifier.predict_proba(x)
        pred = np.argmax(proba, axis=1)
        true_parts.append(y)
        pred_parts.append(pred)
        proba_parts.append(proba.astype(np.float32))

    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(pred_parts)
    y_proba = np.vstack(proba_parts)
    labels = np.arange(len(encoder.classes_))

    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top3_accuracy": float(top_k_accuracy_score(y_true, y_proba, k=3, labels=labels)),
        "top5_accuracy": float(top_k_accuracy_score(y_true, y_proba, k=5, labels=labels)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=encoder.classes_,
            zero_division=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    train_path = find_patient_file(args.dataset_dir, "train")
    validate_path = find_patient_file(args.dataset_dir, "validate")
    test_path = None if args.skip_test else find_patient_file(args.dataset_dir, "test")
    conditions, evidences = load_metadata(args.dataset_dir)

    encoder = LabelEncoder()
    encoder.fit(sorted(conditions.keys()))

    if not DEFAULT_DISEASE_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Thiếu file ánh xạ bệnh DDXPlus: {DEFAULT_DISEASE_MAPPING_PATH}"
        )
    disease_mapping = json.loads(
        DEFAULT_DISEASE_MAPPING_PATH.read_text(encoding="utf-8")
    )
    missing_mappings = sorted(set(encoder.classes_) - set(disease_mapping))
    if missing_mappings:
        raise ValueError(
            "Thiếu ánh xạ tiếng Việt cho các bệnh DDXPlus: "
            + ", ".join(missing_mappings)
        )

    logger.info("Scanning evidence vocabulary from %s", train_path)
    vectorizer = build_evidence_vectorizer(train_path, args.chunksize)
    feature_count = len(vectorizer.vocabulary) + len(AGE_BIN_NAMES) + 4
    class_counts = count_labels(train_path, args.chunksize)
    unknown = sorted(set(class_counts) - set(encoder.classes_))
    if unknown:
        raise ValueError(f"Nhãn PATHOLOGY không có trong release_conditions.json: {unknown}")

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        learning_rate="optimal",
        average=True,
        fit_intercept=True,
        random_state=args.random_state,
        n_jobs=-1,
    )

    logger.info(
        "Training DDXPlus classifier: rows=%s, classes=%s, features=%s, epochs=%s",
        sum(class_counts.values()), len(encoder.classes_), feature_count, args.epochs,
    )
    train_streaming(
        train_path, classifier, encoder, vectorizer,
        class_counts, args.epochs, args.chunksize, args.random_state,
    )

    validation = evaluate(validate_path, classifier, encoder, vectorizer, args.chunksize)
    logger.info(
        "Validation: accuracy=%.4f macro_f1=%.4f top3=%.4f",
        validation["accuracy"], validation["macro_f1"], validation["top3_accuracy"],
    )
    test_metrics = None
    if test_path is not None:
        test_metrics = evaluate(test_path, classifier, encoder, vectorizer, args.chunksize)
        logger.info(
            "Test: accuracy=%.4f macro_f1=%.4f top3=%.4f",
            test_metrics["accuracy"], test_metrics["macro_f1"], test_metrics["top3_accuracy"],
        )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifact_dir / "ddxplus_disease_model.pkl"
    metadata_path = args.artifact_dir / "ddxplus_disease_model_metadata.json"
    report_path = args.artifact_dir / "ddxplus_disease_model_report.txt"

    metadata = {
        "dataset": "DDXPlus Dataset English",
        "dataset_rows": {
            "train": int(sum(class_counts.values())),
            "validation": validation["rows"],
            "test": test_metrics["rows"] if test_metrics else None,
        },
        "algorithm": "Sparse structured evidence features + SGD one-vs-rest logistic classifier",
        "classes": len(encoder.classes_),
        "feature_count": feature_count,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "supports_predict_proba": True,
        "validation": {k: v for k, v in validation.items() if k != "classification_report"},
        "test": {k: v for k, v in (test_metrics or {}).items() if k != "classification_report"},
        "limitations": [
            "DDXPlus is synthetic and English; production use requires clinical validation.",
            "Free Vietnamese text must be mapped to DDXPlus evidence codes before inference.",
            "The output is differential screening support, not a medical diagnosis.",
        ],
    }
    artifact = {
        "model": classifier,
        "label_encoder": encoder,
        "classes": encoder.classes_.tolist(),
        "evidence_vectorizer": vectorizer,
        "disease_mapping": disease_mapping,
        "age_bins": AGE_BINS,
        "age_bin_names": AGE_BIN_NAMES,
        "metadata": metadata,
    }
    joblib.dump(artifact, model_path, compress=3)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "DDXPlus Structured Disease Model Report",
        "=" * 48,
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "\nVALIDATION CLASSIFICATION REPORT\n",
        validation["classification_report"],
    ]
    if test_metrics:
        report_lines.extend(["\nTEST CLASSIFICATION REPORT\n", test_metrics["classification_report"]])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Saved model: %s", model_path)


if __name__ == "__main__":
    main()
