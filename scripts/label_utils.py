import re
from typing import Optional


LABELS = ("yes", "no", "neutral")


def _canonical(label: str) -> str:
    label = label.lower()
    if label == "neither":
        return "neutral"
    return label


def normalize_label(text: str, default: str = "invalid") -> str:
    """Map text to yes/no/neutral; missing or unparsable predictions stay invalid."""
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return default

    cleaned = re.sub(r"^(?:assistant|model|user)\s*[:\-\n]*\s*", "", cleaned)

    marker_option_list = re.search(
        r"\b(?:final\s+answer|answer|decision|label)\s*(?:\([^)]*\))?\s*[:\-]\s*"
        r"yes\s*(?:,|\||/)\s*no\s*(?:,|\||/)?\s*(?:or\s+)?(?:neither|neutral)\b",
        cleaned,
    )
    if marker_option_list:
        return default

    marker_match = re.search(
        r"\b(?:final\s+answer|answer|decision|label)\s*(?:\([^)]*\))?\s*[:\-]\s*"
        r"(yes|no|neither|neutral)\b\s*(?:$|[\r\n.,;]|[-–—]\s)",
        cleaned,
    )
    if marker_match:
        return _canonical(marker_match.group(1))

    sentence_marker_match = re.search(
        r"\b(?:final\s+answer|answer|decision|label)\s+(?:is|would\s+be)\s+"
        r"(yes|no|neither|neutral)\b",
        cleaned,
    )
    if sentence_marker_match:
        return _canonical(sentence_marker_match.group(1))

    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), cleaned)
    first_token_match = re.match(r"^\W*(yes|no|neither|neutral)\b", first_line)
    if first_token_match:
        return _canonical(first_token_match.group(1))

    compact = re.sub(r"\s+", " ", cleaned)
    exact = compact.strip(" .,:;!?()[]{}\"'")
    if exact in {"yes", "no", "neither", "neutral"}:
        return _canonical(exact)

    option_list_pattern = (
        r"\byes\s*(?:,|\||/)\s*no\s*(?:,|\||/)?\s*(?:or\s+)?(?:neither|neutral)\b"
    )
    if re.search(option_list_pattern, compact):
        return default

    if re.search(r"\bneither\b", compact) or re.search(r"\bneutral\b", compact):
        return "neutral"

    # Check negative acceptability before positive acceptability. Otherwise
    # "not socially acceptable" is incorrectly counted as yes.
    negative_patterns = (
        r"\bnot\s+(?:socially\s+)?acceptable\b",
        r"\bnot\s+(?:socially\s+)?appropriate\b",
        r"\bnot\s+considered\s+(?:socially\s+)?acceptable\b",
        r"\b(?:socially\s+)?unacceptable\b",
        r"\binappropriate\b",
    )
    if any(re.search(pattern, compact) for pattern in negative_patterns):
        return "no"

    positive_patterns = (
        r"\b(?:is|are|was|were|be|being|considered)\s+(?:socially\s+)?acceptable\b",
        r"\bsocially\s+acceptable\b",
        r"\bappropriate\b",
    )
    if any(re.search(pattern, compact) for pattern in positive_patterns):
        return "yes"

    return default


def extract_label(text: str, default: str = "invalid") -> str:
    return normalize_label(text, default=default)


def label_or_none(text: str) -> Optional[str]:
    label = normalize_label(text, default="")
    return label if label in LABELS else None
