"""Extract the bundled DDXPlus archive into the training folder.

Run from the project root:
    python scripts/prepare_ddxplus_dataset.py
"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "ddxplus"
ARCHIVE = RAW_DIR / "DDXPlus_Dataset_English.zip"
OUTPUT_DIR = RAW_DIR / "extracted"


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Không tìm thấy archive DDXPlus: {ARCHIVE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ddxplus_") as tmp:
        temp_dir = Path(tmp)
        with zipfile.ZipFile(ARCHIVE) as outer:
            outer.extractall(temp_dir)

        for metadata_name in ["release_conditions.json", "release_evidences.json"]:
            source = temp_dir / metadata_name
            if source.exists():
                shutil.copy2(source, RAW_DIR / metadata_name)

        split_archives = {
            "train": "release_train_patients.zip",
            "validate": "release_validate_patients.zip",
            "test": "release_test_patients.zip",
        }
        for split, filename in split_archives.items():
            source = temp_dir / filename
            if not source.exists():
                raise FileNotFoundError(f"Thiếu {filename} trong archive DDXPlus")
            with zipfile.ZipFile(source) as nested:
                nested.extractall(OUTPUT_DIR)
            print(f"Đã giải nén tập {split}: {filename}")

    print(f"Dữ liệu sẵn sàng tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
