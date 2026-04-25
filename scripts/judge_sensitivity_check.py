import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from label_utils import extract_label, normalize_label


LABELS = ("yes", "no", "neutral", "invalid")


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def scenario_key(row: dict) -> Tuple[str, str, str, str]:
    return (
        str(row.get("Country", "")),
        str(row.get("Story", "")),
        str(row.get("Rule-of-Thumb", "")),
        str(row.get("Gold Label", "")),
    )


def prediction_text(
    row: dict,
    model: str,
    explicit_field: str = "",
    prefer_raw: bool = True,
) -> str:
    if explicit_field:
        return str(row.get(explicit_field, ""))

    raw_field = f"{model}_final_raw"
    final_field = f"{model}_final"

    if prefer_raw and raw_field in row and str(row.get(raw_field, "")).strip():
        return str(row.get(raw_field, ""))

    if final_field in row and str(row.get(final_field, "")).strip():
        return str(row.get(final_field, ""))

    return str(row.get(raw_field, ""))


def judge_base(text: str) -> str:
    return extract_label(text)


def judge_alternate(text: str) -> str:
    return extract_label(text, strict=True)


def accuracy(preds: List[str], gold: List[str]) -> float:
    if not preds:
        return 0.0
    return sum(int(p == g) for p, g in zip(preds, gold)) / len(preds)


def cohen_kappa(a: List[str], b: List[str]) -> float:
    if not a:
        return 0.0
    n = len(a)
    p0 = sum(int(x == y) for x, y in zip(a, b)) / n

    cnt_a = Counter(a)
    cnt_b = Counter(b)
    pe = 0.0
    for label in LABELS:
        pe += (cnt_a[label] / n) * (cnt_b[label] / n)

    if abs(1.0 - pe) < 1e-12:
        return 1.0
    return (p0 - pe) / (1.0 - pe)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge sensitivity check for final predictions.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--alternate_file", default="", help="Optional file with alternate judge outputs for same scenarios.")
    parser.add_argument("--alternate_field", default="", help="If alternate file is set, this field is used as alternate predictions.")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_csv", default="")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_json = (
        Path(args.output_json)
        if args.output_json
        else input_file.with_name(input_file.stem + f"_{args.model}_judge_sensitivity.json")
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else input_file.with_name(input_file.stem + f"_{args.model}_judge_disagreements.csv")
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_file)

    by_key_alt: Dict[Tuple[str, str, str, str], str] = {}
    use_external_alternate = bool(args.alternate_file)

    if use_external_alternate:
        alt_file = Path(args.alternate_file)
        if not alt_file.exists():
            raise FileNotFoundError(f"Alternate file not found: {alt_file}")
        alt_rows = read_jsonl(alt_file)
        for row in alt_rows:
            by_key_alt[scenario_key(row)] = prediction_text(
                row,
                args.model,
                args.alternate_field,
                prefer_raw=bool(args.alternate_field),
            )

    gold: List[str] = []
    pred_base: List[str] = []
    pred_alt: List[str] = []
    disagreements: List[dict] = []

    for idx, row in enumerate(rows):
        base_text = prediction_text(row, args.model)
        g = normalize_label(str(row.get("Gold Label", "")))
        p_base = judge_base(base_text)

        if use_external_alternate:
            alt_text = by_key_alt.get(scenario_key(row), "")
            p_alt = judge_base(alt_text)
        else:
            p_alt = judge_alternate(base_text)

        gold.append(g)
        pred_base.append(p_base)
        pred_alt.append(p_alt)

        if p_base != p_alt:
            disagreements.append(
                {
                    "row_index": idx,
                    "country": row.get("Country", ""),
                    "gold": g,
                    "base_prediction": p_base,
                    "alternate_prediction": p_alt,
                    "raw_prediction_preview": base_text[:180],
                }
            )

    acc_base = accuracy(pred_base, gold)
    acc_alt = accuracy(pred_alt, gold)
    agree = sum(int(x == y) for x, y in zip(pred_base, pred_alt)) / len(pred_base) if pred_base else 0.0
    kappa = cohen_kappa(pred_base, pred_alt)

    summary = {
        "input_file": str(input_file),
        "model": args.model,
        "alternate_source": (
            str(args.alternate_file)
            if use_external_alternate
            else "canonical_strict_parser_on_same_outputs"
        ),
        "base_policy": "canonical_lenient",
        "alternate_policy": "canonical_lenient_from_alternate_file" if use_external_alternate else "canonical_strict",
        "n_rows": len(rows),
        "agreement_rate": round(agree, 6),
        "cohen_kappa": round(kappa, 6),
        "accuracy_base": round(acc_base, 6),
        "accuracy_alternate": round(acc_alt, 6),
        "accuracy_delta_alternate_minus_base": round(acc_alt - acc_base, 6),
        "n_disagreements": len(disagreements),
        "disagreement_rate": round((len(disagreements) / len(rows)) if rows else 0.0, 6),
        "disagreements_csv": str(output_csv),
    }

    write_csv(
        output_csv,
        disagreements,
        [
            "row_index",
            "country",
            "gold",
            "base_prediction",
            "alternate_prediction",
            "raw_prediction_preview",
        ],
    )

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved judge sensitivity summary: {output_json}")
    print(f"Saved disagreement rows: {output_csv}")


if __name__ == "__main__":
    main()
