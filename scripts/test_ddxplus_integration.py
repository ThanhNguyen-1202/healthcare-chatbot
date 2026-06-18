"""Smoke-test the trained DDXPlus model against patient rows."""
from __future__ import annotations
import argparse, ast, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.services.ml_disease_prediction_service import predict_top_diseases  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument(
        "--file",
        type=Path,
        default=ROOT / "data/raw/ddxplus/extracted/release_test_patients",
    )
    args = parser.parse_args()
    if not args.file.exists():
        raise SystemExit(
            f"Không tìm thấy test file: {args.file}\n"
            "Hãy chạy scripts/prepare_ddxplus_dataset.py trước, hoặc truyền --file tới "
            "release_test_patients đã giải nén."
        )
    frame = pd.read_csv(args.file, nrows=args.rows)
    top1 = top3 = 0
    for row in frame.itertuples(index=False):
        codes = ast.literal_eval(row.EVIDENCES)
        ranked = predict_top_diseases(
            evidence_codes=codes,
            age=int(row.AGE),
            gender=str(row.SEX),
            top_k=3,
        )
        canonical = [item["canonical_name"] for item in ranked]
        top1 += int(canonical[0] == row.PATHOLOGY)
        top3 += int(row.PATHOLOGY in canonical)
    total = len(frame)
    print({
        "rows": total,
        "top1_accuracy": round(top1 / total, 6),
        "top3_accuracy": round(top3 / total, 6),
    })


if __name__ == "__main__":
    main()
