

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ViMedical_Disease.csv"

ARTIFACT_DIR = PROJECT_ROOT / "backend" / "app" / "ml" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "disease_model_baseline.pkl"
REPORT_PATH = ARTIFACT_DIR / "disease_model_baseline_report.txt"
METADATA_PATH = ARTIFACT_DIR / "disease_model_baseline_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_SAMPLES_PER_CLASS = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize Vietnamese symptom text before training."""
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    replacements = {
        "\n": " ",
        "\t": " ",
        "\r": " ",
        ",": " ",
        ";": " ",
        ":": " ",
        ".": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        "/": " ",
        "\\": " ",
        "-": " ",
        "_": " ",
        '"': " ",
        "'": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized_columns = {column.lower().strip(): column for column in df.columns}

    for candidate in candidates:
        key = candidate.lower().strip()
        if key in normalized_columns:
            return normalized_columns[key]

    raise ValueError(
        "Không tìm thấy cột phù hợp. "
        f"Cần một trong các cột: {candidates}. "
        f"Các cột hiện có: {list(df.columns)}"
    )


def load_dataset() -> tuple[pd.Series, pd.Series]:
    """Load, clean, and return symptom texts and disease labels."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    logger.info("Loaded dataset: rows=%s", len(df))
    logger.info("Columns: %s", list(df.columns))

    text_column = find_column(
        df,
        [
            "Question",
            "question",
            "questions",
            "symptoms",
            "symptom",
            "trieu_chung",
            "triệu chứng",
            "description",
            "mo_ta",
            "mô tả",
            "text",
            "noi_dung",
            "nội dung",
        ],
    )

    label_column = find_column(
        df,
        [
            "Disease",
            "disease",
            "disease_name",
            "benh",
            "bệnh",
            "label",
            "diagnosis",
            "diagnose",
            "ten_benh",
            "tên bệnh",
        ],
    )

    df = df[[text_column, label_column]].copy()
    df.columns = ["text", "label"]

    df["text"] = df["text"].astype(str).map(normalize_text)
    df["label"] = df["label"].astype(str).str.strip()

    df = df[(df["text"] != "") & (df["label"] != "")]
    df = df.drop_duplicates(subset=["text", "label"])

    label_counts = df["label"].value_counts()
    valid_labels = label_counts[label_counts >= MIN_SAMPLES_PER_CLASS].index
    df = df[df["label"].isin(valid_labels)].copy()

    logger.info("Text column: %s", text_column)
    logger.info("Label column: %s", label_column)
    logger.info("Cleaned dataset rows: %s", len(df))
    logger.info("Number of disease classes: %s", df["label"].nunique())

    if df.empty:
        raise ValueError("Dataset rỗng sau khi làm sạch.")

    if df["label"].nunique() < 2:
        raise ValueError("Dataset cần ít nhất 2 nhãn bệnh để train.")

    return df["text"], df["label"]


def build_model() -> Pipeline:
    """Build model that supports predict_proba().

    Important:
        Do not replace LogisticRegression with LinearSVC here.
        LinearSVC does not support predict_proba(), so FastAPI startup will fail.
    """
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
    )

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
    )

    features = FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_tfidf),
            ("char_tfidf", char_tfidf),
        ],
        n_jobs=-1,
    )

    classifier = LogisticRegression(
        C=4.0,
        class_weight="balanced",
        solver="saga",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("features", features),
            ("classifier", classifier),
        ]
    )


def assert_predict_proba_supported(model: Pipeline) -> None:
    """Fail fast if the trained model cannot produce probabilities."""
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "Model không hỗ trợ predict_proba(). "
            "Hãy dùng LogisticRegression, không dùng LinearSVC."
        )


def evaluate_model(
    model: Pipeline,
    x_test: pd.Series,
    y_test: Any,
    label_encoder: LabelEncoder,
) -> dict[str, Any]:
    """Evaluate trained model and return metrics."""
    assert_predict_proba_supported(model)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    labels = list(range(len(label_encoder.classes_)))

    top3_accuracy = top_k_accuracy_score(
        y_test,
        y_proba,
        k=min(3, len(labels)),
        labels=labels,
    )

    top5_accuracy = top_k_accuracy_score(
        y_test,
        y_proba,
        k=min(5, len(labels)),
        labels=labels,
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "top3_accuracy": float(top3_accuracy),
        "top5_accuracy": float(top5_accuracy),
        "classification_report": report,
    }


def save_artifacts(
    model: Pipeline,
    label_encoder: LabelEncoder,
    metrics: dict[str, Any],
    total_rows: int,
) -> None:
    """Save model artifact, report, and metadata."""
    assert_predict_proba_supported(model)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "algorithm": (
            "Word TF-IDF ngram(1,3) + "
            "Char TF-IDF char_wb(3,5) + "
            "LogisticRegression"
        ),
        "supports_predict_proba": True,
        "accuracy": metrics["accuracy"],
        "top3_accuracy": metrics["top3_accuracy"],
        "top5_accuracy": metrics["top5_accuracy"],
        "rows": total_rows,
        "classes": len(label_encoder.classes_),
        "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "classes": label_encoder.classes_.tolist(),
        "metadata": metadata,
    }

    joblib.dump(artifact, MODEL_PATH)

    loaded_artifact = joblib.load(MODEL_PATH)
    loaded_model = loaded_artifact.get("model", loaded_artifact)

    if not hasattr(loaded_model, "predict_proba"):
        raise RuntimeError(
            f"Saved model at {MODEL_PATH} does not support predict_proba()."
        )

    report_text = "\n".join(
        [
            "Vietnamese Disease Prediction Training Report",
            "=" * 50,
            f"Algorithm: {metadata['algorithm']}",
            f"Supports predict_proba: {metadata['supports_predict_proba']}",
            f"Rows: {metadata['rows']}",
            f"Classes: {metadata['classes']}",
            f"Accuracy: {metadata['accuracy']:.4f}",
            f"Top-3 Accuracy: {metadata['top3_accuracy']:.4f}",
            f"Top-5 Accuracy: {metadata['top5_accuracy']:.4f}",
            "",
            "Classification Report:",
            metrics["classification_report"],
        ]
    )

    REPORT_PATH.write_text(report_text, encoding="utf-8")

    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Saved model: %s", MODEL_PATH)
    logger.info("Saved report: %s", REPORT_PATH)
    logger.info("Saved metadata: %s", METADATA_PATH)
    logger.info("Verified saved model supports predict_proba: True")


def main() -> None:
    """Train disease prediction model."""
    texts, labels = load_dataset()

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        encoded_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=encoded_labels,
    )

    logger.info("Train size: %s", len(x_train))
    logger.info("Test size: %s", len(x_test))

    model = build_model()

    logger.info("Training model...")
    model.fit(x_train, y_train)

    assert_predict_proba_supported(model)

    logger.info("Evaluating model...")
    metrics = evaluate_model(
        model=model,
        x_test=x_test,
        y_test=y_test,
        label_encoder=label_encoder,
    )

    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Top-3 Accuracy: %.4f", metrics["top3_accuracy"])
    logger.info("Top-5 Accuracy: %.4f", metrics["top5_accuracy"])

    save_artifacts(
        model=model,
        label_encoder=label_encoder,
        metrics=metrics,
        total_rows=len(texts),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)
