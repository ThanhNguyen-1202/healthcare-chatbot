import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


INPUT_XML = Path("data/raw/medlineplus_health_topics.xml")
OUTPUT_JSONL = Path("data/processed/medlineplus_rag_knowledge.jsonl")
DISEASE_ALIAS_PATH = Path("backend/app/mappings/disease_aliases.json")


def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def normalize_match_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = html.unescape(text)
    text = re.sub(r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def unique_keep_order(values):
    results = []
    for item in values:
        if item and item not in results:
            results.append(item)
    return results


def extract_all_text(elem) -> str:
    if elem is None:
        return ""

    raw_text = ET.tostring(elem, encoding="unicode", method="xml")
    raw_text = html.unescape(raw_text)
    raw_text = re.sub(r"<[^>]+>", " ", raw_text)
    return clean_text(raw_text)


def load_disease_aliases() -> dict:
    if not DISEASE_ALIAS_PATH.exists():
        return {}

    with open(DISEASE_ALIAS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {}

    clean_data = {}
    for disease_name, aliases in data.items():
        if not isinstance(disease_name, str):
            continue

        if not isinstance(aliases, list):
            aliases = []

        clean_aliases = []
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                clean_aliases.append(alias.strip())

        clean_data[disease_name.strip()] = clean_aliases

    return clean_data


def infer_disease_fields(title: str, alias_mapping: dict) -> tuple[list[str], list[str]]:
    title_norm = normalize_match_text(title)
    matched_disease_names = []
    matched_aliases = [title]

    for disease_name_vn, aliases in alias_mapping.items():
        disease_name_norm = normalize_match_text(disease_name_vn)

        if disease_name_norm and disease_name_norm == title_norm:
            matched_disease_names.append(disease_name_vn)

        for alias in aliases:
            alias_norm = normalize_match_text(alias)

            # Ưu tiên match rất chặt để tránh match sai lung tung
            if alias_norm and (alias_norm == title_norm or title_norm.startswith(alias_norm) or alias_norm.startswith(title_norm)):
                if disease_name_vn not in matched_disease_names:
                    matched_disease_names.append(disease_name_vn)

                if alias not in matched_aliases:
                    matched_aliases.append(alias)

    return unique_keep_order(matched_disease_names), unique_keep_order(matched_aliases)


def infer_department(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()

    # Ưu tiên hô hấp trước da liễu để tránh match sai với vài từ chung chung
    if any(x in text for x in ["pneumonia", "bronchitis", "asthma", "respiratory", "lung", "lungs", "breathing", "shortness of breath", "cough"]):
        return "Hô hấp"

    if any(x in text for x in ["heart", "cardiac", "chest pain", "blood vessel"]):
        return "Tim mạch"

    if any(x in text for x in ["stomach", "abdomen", "digestive", "bowel", "appendix", "gallbladder"]):
        return "Tiêu hóa"

    if any(x in text for x in ["eye", "vision", "glaucoma", "cataract"]):
        return "Mắt"

    if any(x in text for x in ["skin", "rash", "eczema", "dermatitis", "hives", "itching"]):
        return "Da liễu"

    return "Nội tổng quát"


def infer_red_flags(title: str, summary: str) -> list[str]:
    text = f"{title} {summary}".lower()
    flags = []

    candidates = {
        "đau ngực": ["chest pain"],
        "khó thở": ["shortness of breath", "trouble breathing", "difficulty breathing"],
        "ngất": ["fainting", "loss of consciousness"],
        "co giật": ["seizure"],
        "tím tái": ["blue skin", "bluish"],
        "chảy máu nhiều": ["heavy bleeding", "severe bleeding"],
    }

    for vn_flag, keywords in candidates.items():
        if any(keyword in text for keyword in keywords):
            flags.append(vn_flag)

    return unique_keep_order(flags)


def infer_symptoms(title: str, summary: str) -> list[str]:
    text = f"{title} {summary}".lower()
    symptom_map = {
        "sốt": ["fever"],
        "ho": ["cough"],
        "khó thở": ["shortness of breath", "trouble breathing", "difficulty breathing"],
        "đau ngực": ["chest pain"],
        "đau bụng": ["abdominal pain", "stomach pain", "belly pain"],
        "buồn nôn": ["nausea"],
        "đau đầu": ["headache"],
        "mẩn đỏ": ["rash"],
        "ngứa da": ["itching", "itchy skin"],
    }

    found = []
    for vn_symptom, keywords in symptom_map.items():
        if any(keyword in text for keyword in keywords):
            found.append(vn_symptom)

    return unique_keep_order(found)


def main():
    if not INPUT_XML.exists():
        raise FileNotFoundError(f"Không tìm thấy file XML tại: {INPUT_XML}")

    alias_mapping = load_disease_aliases()
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(INPUT_XML)
    root = tree.getroot()

    topics = root.findall(".//health-topic")

    count = 0
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for topic in topics:
            title = clean_text(topic.get("title", ""))

            full_summary_elem = topic.find("full-summary")
            summary = extract_all_text(full_summary_elem)

            if not title or not summary:
                continue

            disease_names, disease_aliases = infer_disease_fields(title, alias_mapping)

            doc = {
                "title": title,
                "disease_names": disease_names,
                "disease_aliases": disease_aliases,
                "symptoms": infer_symptoms(title, summary),
                "department": infer_department(title, summary),
                "red_flags": infer_red_flags(title, summary),
                "content": summary,
                "source": "MedlinePlus"
            }

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

    print("=== MEDLINEPLUS PARSE DONE ===")
    print("INPUT:", INPUT_XML)
    print("OUTPUT:", OUTPUT_JSONL)
    print("TOTAL TOPICS FOUND:", len(topics))
    print("TOTAL DOCS WRITTEN:", count)
    print("ALIAS SOURCE:", DISEASE_ALIAS_PATH)
    print("TOTAL DISEASE ALIAS ENTRIES:", len(alias_mapping))


if __name__ == "__main__":
    main()