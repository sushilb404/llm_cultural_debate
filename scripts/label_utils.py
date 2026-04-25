import re
from typing import Iterable, Optional


LABELS = ("yes", "no", "neutral")
ROLE_NAMES = ("assistant", "model", "user", "system")

_LABEL_VALUE_RE = r"(yes|no|neither|neutral)\b"
_LABEL_LINE_RE = re.compile(
    rf"^(?:final\s+answer|answer|decision|label)\s*(?:\([^)]*\))?\s*[:\-]\s*{_LABEL_VALUE_RE}",
    re.IGNORECASE,
)
_LABEL_MARKER_ONLY_RE = re.compile(
    r"^(?:final\s+answer|answer|decision|label)\s*(?:\([^)]*\))?\s*[:\-]\s*$",
    re.IGNORECASE,
)
_LABEL_SENTENCE_RE = re.compile(
    rf"\b(?:the\s+)?(?:final\s+answer|answer|decision|label)\s+(?:is|would\s+be)\s+{_LABEL_VALUE_RE}",
    re.IGNORECASE,
)
_BARE_LABEL_RE = re.compile(rf"^{_LABEL_VALUE_RE}\s*(?:[.?!])?$", re.IGNORECASE)
_LEADING_LABEL_WITH_TEXT_RE = re.compile(rf"^{_LABEL_VALUE_RE}\s*[:,.?!-]\s*\S", re.IGNORECASE)
_ROLE_ONLY_RE = re.compile(r"^(?:assistant|model|user|system)\s*[:\-]?\s*$", re.IGNORECASE)
_INLINE_ROLE_PREFIX_RE = re.compile(r"^(?:assistant|model|user|system)\s*[:\-]\s*", re.IGNORECASE)


def _canonical(label: str) -> str:
    label = label.lower()
    if label in {"neither", "neutral"}:
        return "neutral"
    return label


def _prepare_text(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _strip_line_role_prefix(line: str) -> str:
    return _INLINE_ROLE_PREFIX_RE.sub("", line.strip())


def _meaningful_lines(text: str) -> Iterable[str]:
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if _ROLE_ONLY_RE.match(stripped):
            continue
        line = _compact_whitespace(_strip_line_role_prefix(stripped))
        if line:
            yield line


def _extract_answer_segment(text: str) -> str:
    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        lowered = lines[index].strip().lower().rstrip(":- ")
        if lowered in {"assistant", "model"}:
            segment = "\n".join(lines[index + 1 :]).strip()
            if segment:
                return segment

    matches = list(re.finditer(r"(?im)^(?:assistant|model)\s*[:\-]\s*(.+)$", text))
    if matches:
        return matches[-1].group(1).strip()

    return text


def _is_option_echo_line(line: str) -> bool:
    compact = _compact_whitespace(line).lower().strip(" .,:;!?")
    if not any(token in compact for token in ("yes", "no", "neither", "neutral")):
        return False
    if "<" in compact and ">" in compact:
        return True
    if "one of" in compact:
        return True
    return all(token in compact for token in ("yes", "no")) and any(
        token in compact for token in ("neither", "neutral")
    )


def _extract_canonical_label(text: str, default: str, strict: bool) -> str:
    cleaned = _prepare_text(text)
    if not cleaned:
        return default

    candidates = []
    answer_segment = _extract_answer_segment(cleaned)
    if answer_segment and answer_segment != cleaned:
        candidates.append(answer_segment)
    candidates.append(cleaned)

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        lines = list(_meaningful_lines(candidate))
        if not lines:
            continue

        for index in range(len(lines) - 1, -1, -1):
            line = lines[index]
            if _is_option_echo_line(line):
                continue
            marker_match = _LABEL_LINE_RE.match(line)
            if marker_match:
                return _canonical(marker_match.group(1))
            if _LABEL_MARKER_ONLY_RE.match(line) and index + 1 < len(lines):
                next_line = lines[index + 1]
                if not _is_option_echo_line(next_line):
                    bare_match = _BARE_LABEL_RE.match(next_line)
                    if bare_match:
                        return _canonical(bare_match.group(1))

        first_line = lines[0]
        if not _is_option_echo_line(first_line):
            bare_match = _BARE_LABEL_RE.match(first_line)
            if bare_match:
                return _canonical(bare_match.group(1))

        if strict:
            continue

        if not _is_option_echo_line(first_line):
            inline_bare_match = _LEADING_LABEL_WITH_TEXT_RE.match(first_line)
            if inline_bare_match:
                return _canonical(inline_bare_match.group(1))

        for line in lines[:3]:
            if _is_option_echo_line(line):
                continue
            sentence_match = _LABEL_SENTENCE_RE.search(line)
            if sentence_match:
                return _canonical(sentence_match.group(1))

        compact = _compact_whitespace(candidate).strip(" .,:;!?()[]{}\"'")
        if compact.lower() in {"yes", "no", "neither", "neutral"}:
            return _canonical(compact)

    return default


def normalize_label(text: str, default: str = "invalid", strict: bool = False) -> str:
    """Map text to yes/no/neutral; missing or unparsable predictions stay invalid."""
    cleaned = _prepare_text(text)
    if not cleaned:
        return default

    canonical = _extract_canonical_label(cleaned, default="", strict=strict)
    if canonical:
        return canonical
    if strict:
        return default

    cleaned = cleaned.lower()
    cleaned = re.sub(r"^(?:assistant|model|user|system)\s*[:\-\n]*\s*", "", cleaned)

    marker_option_list = re.search(
        r"\b(?:final\s+answer|answer|decision|label)\s*(?:\([^)]*\))?\s*[:\-]\s*"
        r"(?:<[^>]*(?:yes|no|neither|neutral)[^>]*>|yes\s*(?:,|\||/)\s*no\s*(?:,|\||/)?\s*"
        r"(?:or\s+)?(?:neither|neutral)\b)",
        cleaned,
    )
    if marker_option_list:
        return default

    compact = _compact_whitespace(cleaned)

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


def extract_label(text: str, default: str = "invalid", strict: bool = False) -> str:
    return normalize_label(text, default=default, strict=strict)


def label_or_none(text: str) -> Optional[str]:
    label = normalize_label(text, default="")
    return label if label in LABELS else None
