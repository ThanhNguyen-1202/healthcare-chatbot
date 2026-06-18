"""Run the Vietnamese DDXPlus holdout benchmark through /predict/ragpp.

This runner is rate-limit aware, retries HTTP 429/5xx responses, saves
checkpoints, and can resume an interrupted result file without rerunning rows
that already completed successfully.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "DDXPlus_Benchmark_1000_Cau_Giong_File_Mau.xlsx"
DEFAULT_MODEL = PROJECT_ROOT / "backend/app/ml/artifacts/ddxplus_vn_text_model.pkl"
DEFAULT_OUTPUT = PROJECT_ROOT / "output/vn_pipeline_holdout_results.xlsx"
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return not str(value).strip()


def _retry_after_seconds(response: requests.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    raw = raw.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(raw)
        now = parsedate_to_datetime(response.headers.get("Date")) if response.headers.get("Date") else None
        if now is not None:
            return max(0.0, (retry_at - now).total_seconds())
    except Exception:
        return None
    return None


def _safe_save(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.stem + ".tmp.xlsx")
    try:
        dataframe.to_excel(temporary_path, index=False, sheet_name="scoring_ready")
        os.replace(temporary_path, output_path)
    except PermissionError as exc:
        temporary_path.unlink(missing_ok=True)
        raise PermissionError(
            f"Không thể lưu {output_path}. Hãy đóng file này trong Excel rồi chạy lại."
        ) from exc


def _normalise_result_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa kiểu dữ liệu của các cột kết quả khi resume từ Excel.

    Pandas mới có thể đọc cột http_status dưới kiểu StringDtype. Khi API trả về
    số nguyên 200, phép gán vào cột chuỗi sẽ phát sinh:
    Invalid value '200' for dtype 'str'.
    """
    defaults: dict[str, Any] = {
        "answer": "",
        "contexts_answer": "[]",
        "ddxplus_evidences": "[]",
        "http_status": "",
        "attempts": 0,
        "elapsed_seconds": 0.0,
        "error": "",
    }

    for column, default in defaults.items():
        if column not in dataframe.columns:
            dataframe[column] = default

    # Các cột có thể chứa chuỗi rỗng, JSON hoặc mã trạng thái.
    # object dtype cho phép resume an toàn từ file Excel có kiểu dữ liệu hỗn hợp.
    for column in (
        "answer",
        "contexts_answer",
        "ddxplus_evidences",
        "http_status",
        "error",
    ):
        dataframe[column] = dataframe[column].astype("object")
        dataframe[column] = dataframe[column].where(
            dataframe[column].notna(), defaults[column]
        )

    dataframe["attempts"] = (
        pd.to_numeric(dataframe["attempts"], errors="coerce")
        .fillna(0)
        .astype("Int64")
    )
    dataframe["elapsed_seconds"] = (
        pd.to_numeric(dataframe["elapsed_seconds"], errors="coerce")
        .fillna(0.0)
        .astype("Float64")
    )
    return dataframe


def _load_benchmark_rows(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    resume: bool,
) -> pd.DataFrame:
    if resume and output_path.exists():
        resumed = pd.read_excel(output_path, sheet_name="scoring_ready")
        if "question" not in resumed.columns:
            raise ValueError(f"File kết quả {output_path} không có cột question")
        print(f"Resume từ file đã có: {output_path} ({len(resumed)} dòng)")
        return _normalise_result_columns(resumed)

    source = pd.read_excel(input_path, sheet_name="scoring_ready")
    artifact = joblib.load(model_path)
    indices = artifact.get("test_indices")
    if indices is None:
        raise KeyError(f"Model artifact {model_path} không có test_indices")

    test_data = source.iloc[indices].copy().reset_index(drop=True)
    for column, default in {
        "answer": "",
        "contexts_answer": "[]",
        "ddxplus_evidences": "[]",
        "http_status": "",
        "attempts": 0,
        "elapsed_seconds": 0.0,
        "error": "",
    }.items():
        if column not in test_data.columns:
            test_data[column] = default
    return _normalise_result_columns(test_data)


def _post_with_retry(
    session: requests.Session,
    api_url: str,
    payload: dict[str, Any],
    timeout: float,
    max_retries: int,
    retry_base: float,
) -> tuple[requests.Response, int, float]:
    started = time.perf_counter()
    last_response: requests.Response | None = None

    for attempt in range(1, max_retries + 2):
        try:
            response = session.post(api_url, json=payload, timeout=timeout)
            last_response = response

            if response.status_code not in TRANSIENT_STATUS_CODES:
                response.raise_for_status()
                return response, attempt, time.perf_counter() - started

            if attempt > max_retries:
                response.raise_for_status()

            retry_after = _retry_after_seconds(response)
            if response.status_code == 429:
                # SlowAPI may omit Retry-After. A longer exponential wait lets
                # the one-minute window expire instead of hammering the API.
                fallback_wait = min(65.0, max(5.0, retry_base * (2 ** (attempt - 1))))
            else:
                fallback_wait = min(30.0, max(1.0, retry_base * (2 ** (attempt - 1))))

            wait_seconds = max(retry_after or 0.0, fallback_wait)
            wait_seconds += random.uniform(0.0, 0.35)
            print(
                f"  HTTP {response.status_code}; thử lại {attempt}/{max_retries + 1} "
                f"sau {wait_seconds:.1f}s"
            )
            time.sleep(wait_seconds)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt > max_retries:
                raise
            wait_seconds = min(30.0, max(1.0, retry_base * (2 ** (attempt - 1))))
            wait_seconds += random.uniform(0.0, 0.35)
            print(f"  Lỗi kết nối: {exc}; thử lại sau {wait_seconds:.1f}s")
            time.sleep(wait_seconds)

    if last_response is not None:
        last_response.raise_for_status()
    raise RuntimeError("Không nhận được phản hồi từ API")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark API tiếng Việt có retry 429 và checkpoint."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/predict/ragpp")
    parser.add_argument(
        "--delay",
        type=float,
        default=2.2,
        help="Nghỉ giữa hai câu. 2.2 giây phù hợp PREDICT_RATE_LIMIT=30/minute.",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-base", type=float, default=5.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tiếp tục file output hiện có và chỉ chạy lại dòng trống/lỗi.",
    )
    args = parser.parse_args()

    if args.delay < 2.0:
        print(
            "CẢNH BÁO: delay < 2 giây dễ vượt PREDICT_RATE_LIMIT=30/minute. "
            "Chỉ dùng delay thấp khi đã tăng giới hạn trong backend/.env và restart backend."
        )

    test_data = _load_benchmark_rows(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        resume=args.resume,
    )

    test_data = _normalise_result_columns(test_data)

    indices_to_run: list[int] = []
    for index, row in test_data.iterrows():
        completed = not _is_blank(row.get("answer")) and _is_blank(row.get("error"))
        if args.resume and completed:
            continue
        indices_to_run.append(index)

    if args.limit is not None:
        indices_to_run = indices_to_run[: max(0, args.limit)]

    print(
        f"Tổng dòng: {len(test_data)} | cần chạy: {len(indices_to_run)} | "
        f"delay={args.delay}s | max_retries={args.max_retries}"
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "DDXPlus-VN-Benchmark/2.1"})
    processed_since_save = 0

    for position, index in enumerate(indices_to_run, start=1):
        row = test_data.loc[index]
        question = str(row.get("question", "")).strip()
        if not question:
            test_data.at[index, "error"] = "Câu hỏi trống"
            continue

        try:
            response, attempts, elapsed = _post_with_retry(
                session=session,
                api_url=args.api_url,
                payload={
                    "message": question,
                    "top_k": 1,
                    "prediction_mode": "hybrid",
                    "use_llm_extractor": False,
                },
                timeout=args.timeout,
                max_retries=max(0, args.max_retries),
                retry_base=max(0.5, args.retry_base),
            )
            body = response.json()
            payload = body.get("data", body)
            top = payload.get("top_diseases", []) or []

            test_data.at[index, "answer"] = top[0].get("name", "") if top else ""
            test_data.at[index, "contexts_answer"] = json.dumps(
                payload.get("ragpp", {}).get("evidence", []) or [],
                ensure_ascii=False,
            )
            test_data.at[index, "ddxplus_evidences"] = json.dumps(
                payload.get("ddxplus_evidences", []) or [],
                ensure_ascii=False,
            )
            test_data.at[index, "http_status"] = str(response.status_code)
            test_data.at[index, "attempts"] = attempts
            test_data.at[index, "elapsed_seconds"] = round(elapsed, 3)
            test_data.at[index, "error"] = ""
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "")
            test_data.at[index, "answer"] = ""
            test_data.at[index, "contexts_answer"] = "[]"
            test_data.at[index, "ddxplus_evidences"] = "[]"
            test_data.at[index, "http_status"] = str(status) if status else ""
            test_data.at[index, "error"] = str(exc)

        processed_since_save += 1
        print(
            f"{position}/{len(indices_to_run)} | excel_row={index + 2} | "
            f"answer={test_data.at[index, 'answer']} | "
            f"status={test_data.at[index, 'http_status']} | "
            f"error={test_data.at[index, 'error']}"
        )

        if processed_since_save >= max(1, args.save_every):
            _safe_save(test_data, args.output)
            processed_since_save = 0

        if position < len(indices_to_run):
            time.sleep(max(args.delay, 0.0))

    _safe_save(test_data, args.output)

    successful = int(
        sum(
            not _is_blank(value)
            for value in test_data["answer"].tolist()
        )
    )
    failed = int(
        sum(
            not _is_blank(value)
            for value in test_data["error"].tolist()
        )
    )
    print(f"Đã lưu: {args.output}")
    print(f"Hoàn thành: {successful}/{len(test_data)} | còn lỗi: {failed}")


if __name__ == "__main__":
    main()
