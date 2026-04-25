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


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def word_count(text: str) -> int:
    return len((text or "").split())


def infer_prediction_field(sample: dict, model_name: Optional[str]) -> str:
    if model_name:
        if f"{model_name}_final" in sample:
            return f"{model_name}_final"
        if model_name in sample:
            return model_name

    finals = [key for key in sample if key.endswith("_final")]
    if finals:
        return finals[0]

    candidates = [key for key in sample if key not in REQUIRED_KEYS and key not in {"error", "model_id"}]
    if candidates:
        return candidates[0]

    raise ValueError("Could not infer a prediction field from the input sample.")


def analyze_file(rows: List[dict], model_name: str, prediction_field: Optional[str] = None) -> Tuple[dict, List[dict], List[str]]:
    if not rows:
        raise ValueError("Input file has no rows.")

    field = prediction_field if prediction_field else infer_prediction_field(rows[0], model_name)
    has_turns = any(key.endswith("_1") for key in rows[0]) and any(key.endswith("_final") for key in rows[0])
    prefix = model_name if f"{model_name}_final" in rows[0] else field[:-6] if field.endswith("_final") else model_name

    golds: List[str] = []
    preds: List[str] = []
    label_counts = Counter()
    country_correct: Dict[str, int] = defaultdict(int)
    country_total: Dict[str, int] = defaultdict(int)
    stance_changed = 0
    neutral_collapse = 0
    total = 0

    story_words: List[int] = []
    rot_words: List[int] = []
    pred_words: List[int] = []
    turn1_words: List[int] = []
    turn2_words: List[int] = []
    final_words: List[int] = []
    interaction_words: List[int] = []

    for row in rows:
        gold = classify_label(str(row.get("Gold Label", "")))
        pred = classify_label(str(row.get(field, "")))
        country = str(row.get("Country", "unknown"))

        total += 1
        golds.append(gold)
        preds.append(pred)
        label_counts[pred] += 1
        country_total[country] += 1
        if pred == gold:
            country_correct[country] += 1

        story_words.append(word_count(str(row.get("Story", ""))))
        rot_words.append(word_count(str(row.get("Rule-of-Thumb", ""))))
        pred_words.append(word_count(str(row.get(field, ""))))

        if has_turns:
            initial = classify_label(str(row.get(f"{prefix}_1", "")))
            final = classify_label(str(row.get(f"{prefix}_final", "")))
            feedback = str(row.get(f"{prefix}_2", ""))
            turn1_len = word_count(str(row.get(f"{prefix}_1", "")))
            turn2_len = word_count(feedback)
            final_len = word_count(str(row.get(f"{prefix}_final", "")))
            turn1_words.append(turn1_len)
            turn2_words.append(turn2_len)
            final_words.append(final_len)
            interaction_words.append(turn1_len + turn2_len + final_len)
            if initial != final:
                stance_changed += 1
            if initial != "neutral" and final == "neutral":
                neutral_collapse += 1

    avg_story_words = sum(story_words) / len(story_words) if story_words else 0.0
    avg_rot_words = sum(rot_words) / len(rot_words) if rot_words else 0.0
    avg_pred_words = sum(pred_words) / len(pred_words) if pred_words else 0.0
    avg_initial_words = sum(turn1_words) / len(turn1_words) if turn1_words else 0.0
    avg_feedback_words = sum(turn2_words) / len(turn2_words) if turn2_words else 0.0
    avg_final_words = sum(final_words) / len(final_words) if final_words else 0.0
    avg_interaction_words = sum(interaction_words) / len(interaction_words) if interaction_words else avg_pred_words

    per_country_rows = []
    for country in sorted(country_total):
        country_acc = country_correct[country] / country_total[country] if country_total[country] else 0.0
        per_country_rows.append(
            {
                "model": model_name,
                "country": country,
                "total": country_total[country],
                "correct": country_correct[country],
                "accuracy": round(country_acc, 6),
            }
        )

    country_acc_values = [row["accuracy"] for row in per_country_rows]
    summary = {
        "model": model_name,
        "prediction_field": field,
        "total": total,
        "accuracy": round((sum(int(p == g) for p, g in zip(preds, golds)) / total) if total else 0.0, 6),
        "label_distribution": {label: label_counts[label] for label in ["yes", "no", "neutral"]},
        "label_rates": {label: round(label_counts[label] / total, 6) if total else 0.0 for label in ["yes", "no", "neutral"]},
        "avg_story_words": round(avg_story_words, 6),
        "avg_rot_words": round(avg_rot_words, 6),
        "avg_prediction_words": round(avg_pred_words, 6),
        "country_accuracy_min": round(min(country_acc_values) if country_acc_values else 0.0, 6),
        "country_accuracy_max": round(max(country_acc_values) if country_acc_values else 0.0, 6),
        "country_accuracy_gap": round((max(country_acc_values) - min(country_acc_values)) if country_acc_values else 0.0, 6),
        "has_turns": has_turns,
        "stance_changed_rate": round((stance_changed / total) if total and has_turns else 0.0, 6),
        "neutral_collapse_rate": round((neutral_collapse / total) if total and has_turns else 0.0, 6),
        "avg_initial_words": round(avg_initial_words, 6),
        "avg_feedback_words": round(avg_feedback_words, 6),
        "avg_final_words": round(avg_final_words, 6),
        "avg_interaction_words": round(avg_interaction_words, 6),
        "interaction_load_ratio": round((sum(interaction_words) / sum(story_words)) if interaction_words and sum(story_words) else 0.0, 6),
        "warning_yes_only": bool(total and label_counts["yes"] == total),
    }

    return summary, per_country_rows, preds


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze model interaction load and AI-to-AI agreement.")
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--model_names", nargs="*", default=None)
    parser.add_argument("--prediction_fields", nargs="*", default=None)
    parser.add_argument("--output_dir", default="results/analysis_ai_interactions")
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
    country_rows = []
    predictions_by_model: Dict[str, List[str]] = {}

    for model_name, input_file, field in zip(model_names, input_files, prediction_fields):
        rows = read_jsonl(input_file)
        summary, per_country_rows, preds = analyze_file(rows, model_name, field)
        summary["input_file"] = str(input_file)
        summaries.append(summary)
        country_rows.extend(per_country_rows)
        predictions_by_model[model_name] = preds

    pairwise_rows = []
    model_list = list(predictions_by_model)
    for i, model_a in enumerate(model_list):
        for model_b in model_list[i + 1 :]:
            preds_a = predictions_by_model[model_a]
            preds_b = predictions_by_model[model_b]
            pairwise_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "agreement": round(agreement(preds_a, preds_b), 6),
                    "kappa": round(cohen_kappa(preds_a, preds_b), 6),
                }
            )

    summary = {
        "models": summaries,
        "pairwise": pairwise_rows,
        "highest_accuracy_model": max(summaries, key=lambda item: item["accuracy"]) ["model"] if summaries else "",
        "highest_yes_rate_model": max(summaries, key=lambda item: item["label_rates"]["yes"]) ["model"] if summaries else "",
        "highest_load_model": max(summaries, key=lambda item: item["avg_interaction_words"]) ["model"] if summaries else "",
        "yes_only_warning": any(item["warning_yes_only"] for item in summaries),
    }

    summary_path = output_dir / "ai_interactions_summary.json"
    pairwise_path = output_dir / "ai_interactions_pairwise.csv"
    country_path = output_dir / "ai_interactions_country_metrics.csv"

    write_json(summary_path, summary)
    write_csv(pairwise_path, pairwise_rows, ["model_a", "model_b", "agreement", "kappa"])
    write_csv(country_path, country_rows, ["model", "country", "total", "correct", "accuracy"])

    print(f"Saved AI interaction summary: {summary_path}")
    if summary["yes_only_warning"]:
        print("Warning: at least one model is yes-only or effectively yes-only.")


if __name__ == "__main__":
    main()
