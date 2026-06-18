import json
import re
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZIP_FILE = PROJECT_ROOT / "data" / "raw" / "Corpus_Redone.zip"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "rag_corpus_redone.jsonl"


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_title(filename: str) -> str:
    name = Path(filename).stem
    name = name.replace("-", " ").replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()


def split_into_chunks(text: str, max_chars: int = 1200) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            sentences = re.split(r"(?<=[.!?。])\s+", paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current) + len(sentence) + 1 <= max_chars:
                    current = f"{current} {sentence}".strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sentence
            continue

        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def infer_topic_type(title: str, content: str) -> str:
    text = f"{title} {content}".lower()

    if any(word in text for word in ["triệu chứng", "dấu hiệu", "biểu hiện"]):
        return "symptoms"

    if any(word in text for word in ["điều trị", "chữa trị", "thuốc", "phác đồ"]):
        return "treatment"

    if any(word in text for word in ["nguyên nhân", "yếu tố nguy cơ"]):
        return "causes"

    return "medical_knowledge"


def build_record(filename: str, chunk: str, index: int, total: int) -> dict:
    title = normalize_title(filename)
    topic_type = infer_topic_type(title, chunk)

    return {
        "id": f"corpus_redone::{Path(filename).stem}::{index}",
        "title": title,
        "source": "corpus_redone_txt",
        "source_file": filename,
        "language": "vi",
        "content": chunk,
        "text": chunk,
        "metadata": {
            "source_type": "txt_zip",
            "topic_type": topic_type,
            "chunk_index": index,
            "total_chunks": total,
        },
    }


def main():
    if not ZIP_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file ZIP: {ZIP_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    records = []
    total_txt_files = 0

    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        txt_names = [
            name for name in zip_ref.namelist()
            if name.lower().endswith(".txt") and not name.endswith("/")
        ]

        for filename in sorted(txt_names):
            raw_bytes = zip_ref.read(filename)
            raw_text = raw_bytes.decode("utf-8", errors="ignore")
            text = clean_text(raw_text)

            if not text:
                continue

            chunks = split_into_chunks(text)
            total_txt_files += 1

            for index, chunk in enumerate(chunks, start=1):
                records.append(build_record(filename, chunk, index, len(chunks)))

            print(f"Đã xử lý {filename}: {len(chunks)} chunks")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nHoàn tất.")
    print(f"Tổng file TXT: {total_txt_files}")
    print(f"Tổng chunks: {len(records)}")
    print(f"File xuất ra: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()