from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAIN_RAG_FILE = PROJECT_ROOT / "data" / "processed" / "rag_knowledge_base_merged.jsonl"
CORPUS_RAG_FILE = PROJECT_ROOT / "data" / "processed" / "rag_corpus_redone.jsonl"
BACKUP_FILE = PROJECT_ROOT / "data" / "processed" / "rag_knowledge_base_merged.backup_before_corpus.jsonl"


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def remove_old_corpus_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if '"source": "corpus_redone_txt"' not in line]


def main():
    if not MAIN_RAG_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file RAG chính: {MAIN_RAG_FILE}")

    if not CORPUS_RAG_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file corpus RAG: {CORPUS_RAG_FILE}")

    main_lines = read_lines(MAIN_RAG_FILE)
    corpus_lines = read_lines(CORPUS_RAG_FILE)

    BACKUP_FILE.write_text("\n".join(main_lines) + "\n", encoding="utf-8")

    cleaned_main_lines = remove_old_corpus_lines(main_lines)
    merged_lines = cleaned_main_lines + corpus_lines

    MAIN_RAG_FILE.write_text("\n".join(merged_lines) + "\n", encoding="utf-8")

    print("Đã merge corpus vào RAG chính.")
    print(f"Backup: {BACKUP_FILE}")
    print(f"Số dòng RAG cũ: {len(main_lines)}")
    print(f"Số dòng RAG sau khi bỏ corpus cũ: {len(cleaned_main_lines)}")
    print(f"Số dòng corpus thêm: {len(corpus_lines)}")
    print(f"Tổng số dòng mới: {len(merged_lines)}")
    print(f"File RAG chính: {MAIN_RAG_FILE}")


if __name__ == "__main__":
    main()