import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict


MAPPINGS_DIR = Path(__file__).resolve().parent.parent / "mappings"


@lru_cache(maxsize=32)
def load_json_mapping(filename: str) -> Dict[str, Any]:
    path = MAPPINGS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required mapping file not found: {path}")

    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if not isinstance(payload, dict):
        raise ValueError(f"Mapping file must contain a JSON object: {path}")
    return payload
