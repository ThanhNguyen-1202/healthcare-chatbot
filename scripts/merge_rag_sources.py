from pathlib import Path

INTERNAL_PATH = Path("data/processed/rag_knowledge_base.jsonl")
MEDLINEPLUS_PATH = Path("data/processed/medlineplus_rag_knowledge.jsonl")
OUTPUT_PATH = Path("data/processed/rag_knowledge_base_merged.jsonl")


def read_lines(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    internal_lines = read_lines(INTERNAL_PATH)
    medlineplus_lines = read_lines(MEDLINEPLUS_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for line in internal_lines:
            f.write(line + "\n")
        for line in medlineplus_lines:
            f.write(line + "\n")

    print("=== MERGE DONE ===")
    print("INTERNAL DOCS:", len(internal_lines))
    print("MEDLINEPLUS DOCS:", len(medlineplus_lines))
    print("TOTAL MERGED DOCS:", len(internal_lines) + len(medlineplus_lines))
    print("OUTPUT:", OUTPUT_PATH)


if __name__ == "__main__":
    main()