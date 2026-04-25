import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.label_utils import classify_label


REQUIRED_KEYS = {"Country", "Story", "Rule-of-Thumb", "Gold Label"}


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_prediction_field(sample: dict) -> str:
    candidate_fields = [key for key in sample if key not in REQUIRED_KEYS and key not in {"error", "model_id"}]
    finals = [key for key in candidate_fields if key.endswith("_final")]
    if finals:
        return finals[0]
    if len(candidate_fields) == 1:
        return candidate_fields[0]
    if candidate_fields:
        return candidate_fields[0]
    raise ValueError("Could not infer prediction field from input file.")


def scenario_key(row: dict) -> Tuple[str, str, str, str]:
    return (
        str(row.get("Country", "")),
        str(row.get("Story", "")),
        str(row.get("Rule-of-Thumb", "")),
        str(row.get("Gold Label", "")),
    )


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def agreement(a: List[str], b: List[str]) -> float:
    if not a:
        return 0.0
    return sum(int(x == y) for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: List[str], b: List[str]) -> float:
    labels = ("yes", "no", "neutral")
    if not a:
        return 0.0
    n = len(a)
    p0 = agreement(a, b)
    cnt_a = Counter(a)
    cnt_b = Counter(b)
    pe = 0.0
    for label in labels:
        pe += (cnt_a[label] / n) * (cnt_b[label] / n)
    if abs(1.0 - pe) < 1e-12:
        return 1.0
    return (p0 - pe) / (1.0 - pe)


def summarize_file(path: Path, model_name: str, output_field: Optional[str] = None) -> Tuple[dict, List[str], List[str], List[str], List[dict]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Input file has no rows: {path}")

    field = output_field if output_field else infer_prediction_field(rows[0])

    golds: List[str] = []
    preds: List[str] = []
    countries: List[str] = []
    label_counts = Counter()
    per_country_correct: Dict[str, int] = defaultdict(int)
    per_country_total: Dict[str, int] = defaultdict(int)
    changed_count = 0
    total = 0
    correct = 0
    yes_only_rows = 0

    for row in rows:
        gold = classify_label(str(row.get("Gold Label", "")))
        pred = classify_label(str(row.get(field, "")))
        country = str(row.get("Country", "unknown"))

        total += 1
        label_counts[pred] += 1
        per_country_total[country] += 1
        golds.append(gold)
        preds.append(pred)
        countries.append(country)

        if pred == gold:
            correct += 1
            per_country_correct[country] += 1

        if pred == "yes" and gold != "yes":
            yes_only_rows += 1

        if field.endswith("_final"):
            initial_field = field[:-6] + "_1"
            if classify_label(str(row.get(initial_field, ""))) != pred:
                changed_count += 1

    yes_rate = (label_counts["yes"] / total) if total else 0.0
    no_rate = (label_counts["no"] / total) if total else 0.0
    neutral_rate = (label_counts["neutral"] / total) if total else 0.0
    accuracy = (correct / total) if total else 0.0

    country_acc = {
        country: (per_country_correct[country] / per_country_total[country]) if per_country_total[country] else 0.0
        for country in per_country_total
    }

    summary = {
        "model": model_name,
        "input_file": str(path),
        "prediction_field": field,
        "total": total,
        "accuracy": round(accuracy, 6),
        "label_distribution": {
            "yes": label_counts["yes"],
            "no": label_counts["no"],
            "neutral": label_counts["neutral"],
        },
        "label_rates": {
            "yes": round(yes_rate, 6),
            "no": round(no_rate, 6),
            "neutral": round(neutral_rate, 6),
        },
        "yes_only_rows": yes_only_rows,
        "yes_only_rate": round((yes_only_rows / total) if total else 0.0, 6),
        "country_accuracy_min": round(min(country_acc.values()) if country_acc else 0.0, 6),
        "country_accuracy_max": round(max(country_acc.values()) if country_acc else 0.0, 6),
        "country_accuracy_gap": round((max(country_acc.values()) - min(country_acc.values())) if country_acc else 0.0, 6),
        "stance_changed": int(changed_count),
        "stance_changed_rate": round((changed_count / total) if total else 0.0, 6),
        "warning_yes_only": bool(no_rate == 0.0 and neutral_rate == 0.0 and yes_rate > 0.0),
    }

    per_country_rows = []
    for country in sorted(per_country_total):
        total_country = per_country_total[country]
        correct_country = per_country_correct[country]
        per_country_rows.append(
            {
                "model": model_name,
                "country": country,
                "total": total_country,
                "correct": correct_country,
                "accuracy": round((correct_country / total_country) if total_country else 0.0, 6),
            }
        )

    return summary, golds, preds, countries, per_country_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model role across multiple output files.")
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--model_names", nargs="*", default=None)
    parser.add_argument("--output_dir", default="results/analysis_model_role")
    parser.add_argument("--prediction_fields", nargs="*", default=None)
    args = parser.parse_args()

    input_files = [Path(path) for path in args.input_files]
    for path in input_files:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    model_names = args.model_names if args.model_names else [path.stem for path in input_files]
    if len(model_names) != len(input_files):
        raise ValueError("--model_names must match --input_files length when provided.")

    prediction_fields = args.prediction_fields if args.prediction_fields else [None] * len(input_files)
    if len(prediction_fields) != len(input_files):
        raise ValueError("--prediction_fields must match --input_files length when provided.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    aligned_predictions: Dict[str, List[str]] = {}
    aligned_golds: Optional[List[str]] = None
    aligned_countries: Optional[List[str]] = None
    country_rows: List[dict] = []

    for model_name, input_file, field in zip(model_names, input_files, prediction_fields):
        summary, golds, preds, countries, per_country_rows = summarize_file(input_file, model_name, field)
        summaries.append(summary)
        aligned_predictions[model_name] = preds
        if aligned_golds is None:
            aligned_golds = golds
        if aligned_countries is None:
            aligned_countries = countries
        country_rows.extend(per_country_rows)

    pairwise_rows: List[dict] = []
    model_list = list(aligned_predictions)
    for i, model_a in enumerate(model_list):
        for model_b in model_list[i + 1 :]:
            preds_a = aligned_predictions[model_a]
            preds_b = aligned_predictions[model_b]
            pairwise_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "agreement": round(agreement(preds_a, preds_b), 6),
                    "kappa": round(cohen_kappa(preds_a, preds_b), 6),
                }
            )

    comparison = {
        "input_files": [str(path) for path in input_files],
        "models": summaries,
        "pairwise": pairwise_rows,
        "top_yes_rate_model": max(summaries, key=lambda item: item["label_rates"]["yes"])["model"] if summaries else "",
        "top_accuracy_model": max(summaries, key=lambda item: item["accuracy"])["model"] if summaries else "",
        "any_yes_only_warning": any(item["warning_yes_only"] for item in summaries),
    }

    summary_path = output_dir / "model_role_summary.json"
    pairwise_path = output_dir / "model_role_pairwise.csv"
    country_path = output_dir / "model_role_country_metrics.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    write_csv(pairwise_path, pairwise_rows, ["model_a", "model_b", "agreement", "kappa"])
    write_csv(country_path, country_rows, ["model", "country", "total", "correct", "accuracy"])

    print(f"Saved model role summary: {summary_path}")
    if comparison["any_yes_only_warning"]:
        print("Warning: at least one model is yes-only or effectively yes-only.")


if __name__ == "__main__":
    main()
