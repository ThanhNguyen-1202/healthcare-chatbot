
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from app.services.ddxplus_evidence_catalog import normalize_evidence_codes


NEGATION_MARKERS = (
    "không", "khong", "chưa", "chua", "không có", "khong co",
    "không bị", "khong bi", "không thấy", "khong thay", "phủ nhận",
    "không ghi nhận", "chưa từng", "chưa bao giờ",
)

# Only binary evidence codes are mapped here.  Categorical/multivalue evidence
# stays with the constrained LLM because selecting a wrong value can hurt more
# than omitting it.
VN_EVIDENCE_ALIASES: Dict[str, Sequence[str]] = {
    "E_91": ("sốt", "sốt cao", "nóng sốt", "phát sốt", "ớn lạnh kèm sốt"),
    "E_201": ("ho", "ho khan", "ho nhiều", "cơn ho"),
    "E_77": ("ho có đờm", "khạc đờm", "đờm vàng", "đờm xanh", "nhiều đờm"),
    "E_203": ("ho thành cơn", "cơn ho dữ dội", "ho rũ rượi"),
    "E_45": ("ho ra máu", "khạc ra máu"),
    "E_66": ("khó thở", "hụt hơi", "thở không nổi", "ngộp thở", "thở gấp"),
    "E_67": ("khó thở về đêm", "thức giấc vì khó thở", "ngộp thở ban đêm"),
    "E_214": ("khò khè", "thở khò khè", "tiếng rít khi thở ra"),
    "E_97": ("đau họng", "rát họng", "viêm họng"),
    "E_181": ("nghẹt mũi", "sổ mũi", "chảy nước mũi trong"),
    "E_182": ("dịch mũi vàng", "dịch mũi xanh", "mũi có mủ"),
    "E_155": ("tim đập nhanh", "hồi hộp", "đánh trống ngực", "nhịp tim không đều"),
    "E_82": ("chóng mặt", "choáng váng", "xây xẩm", "sắp ngất", "muốn ngất"),
    "E_148": ("buồn nôn", "muốn nôn", "nôn nao"),
    "E_211": ("nôn nhiều", "nôn liên tục", "ói nhiều", "mửa nhiều"),
    "E_51": ("tiêu chảy", "đi ngoài nhiều", "đi phân lỏng"),
    "E_178": ("chảy máu bất thường", "dễ bầm", "bầm tím bất thường", "xuất huyết"),
    "E_151": ("sưng", "phù", "sưng phù", "sưng khu trú"),
    "E_52": ("nhìn đôi", "song thị", "thấy hai hình"),
    "E_172": ("sụp mí", "khó mở mắt", "mí mắt rũ"),
    "E_90": ("yếu tăng khi hoạt động", "yếu cơ tăng khi mệt", "yếu cơ cuối ngày"),
    "E_83": ("yếu cơ mặt", "yếu cơ mắt", "mặt yếu"),
    "E_84": ("yếu hai tay", "yếu hai chân", "yếu cả tay chân"),
    "E_157": ("tê quanh miệng", "tê cả tay chân", "tê hai tay hai chân"),
    "E_177": ("tê bì", "mất cảm giác", "châm chích", "kiến bò"),
    "E_81": ("lo âu kéo dài", "lo lắng kéo dài", "lo âu mạn tính"),
    "E_50": ("vã mồ hôi", "đổ mồ hôi nhiều", "toát mồ hôi"),
    "E_162": ("sụt cân", "giảm cân không chủ ý", "gầy sút"),
    "E_65": ("khó nuốt", "nuốt vướng", "nuốt nghẹn"),
    "E_212": ("khàn tiếng", "giọng khàn", "mất tiếng"),
    "E_14": ("đau ngực khi nghỉ", "đau ngực cả lúc nghỉ", "đau ngực không giảm khi nghỉ"),
    "E_173": ("ợ nóng", "nóng rát thượng vị", "trào ngược", "vị chua trong miệng"),
}


@dataclass(frozen=True)
class EvidenceMatch:
    code: str
    alias: str
    negated: bool
    start: int


def normalize_vietnamese_text(value: str) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_negated(text: str, start: int) -> bool:
    window = text[max(0, start - 45):start]
    # A contrast marker starts a new clause, so a negation before it should not
    # affect the current symptom.
    for marker in (" nhưng ", " tuy nhiên ", ";", ".", "!"):
        if marker in window:
            window = window.rsplit(marker, 1)[-1]
    return any(marker in window for marker in NEGATION_MARKERS)


def extract_rule_evidence_matches(text: str) -> List[EvidenceMatch]:
    normalized = normalize_vietnamese_text(text)
    matches: List[EvidenceMatch] = []
    seen = set()
    for code, aliases in VN_EVIDENCE_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            start = normalized.find(alias)
            if start < 0:
                continue
            key = (code, start)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                EvidenceMatch(
                    code=code,
                    alias=alias,
                    negated=_is_negated(normalized, start),
                    start=start,
                )
            )
            break
    matches.sort(key=lambda item: item.start)
    return matches


def extract_rule_evidences(text: str) -> Dict[str, List[str]]:
    matches = extract_rule_evidence_matches(text)
    positive: List[str] = []
    negative: List[str] = []
    for match in matches:
        target = negative if match.negated else positive
        if match.code not in target:
            target.append(match.code)
    return {
        "positive": normalize_evidence_codes(positive),
        "negative": normalize_evidence_codes(negative),
        "matched_aliases": [
            f"{'NEG' if item.negated else 'POS'}:{item.alias}->{item.code}"
            for item in matches
        ],
    }


def merge_hybrid_evidences(text: str, llm_codes: Iterable[str] | None) -> Dict[str, List[str]]:
    rule_result = extract_rule_evidences(text)
    positive_rules = rule_result["positive"]
    negative_rules = rule_result["negative"]

    merged = normalize_evidence_codes(list(llm_codes or []) + positive_rules)
    negative_bases = {item.split("_@_", 1)[0] for item in negative_rules}
    filtered = [
        item for item in merged
        if item.split("_@_", 1)[0] not in negative_bases
    ]
    return {
        "positive": filtered,
        "negative": negative_rules,
        "rule_positive": positive_rules,
        "matched_aliases": rule_result["matched_aliases"],
    }
