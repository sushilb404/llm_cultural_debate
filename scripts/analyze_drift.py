import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


LABELS = {"yes", "no", "neutral"}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_label(text: str) -> str:
    cleaned = (text or "").replace(",", "").replace(".", "").strip().lower()
    if "yes" in cleaned or "socially acceptable" in cleaned:
        return "yes"
    if "no" in cleaned or "not socially acceptable" in cleaned:
        return "no"
    if "neither" in cleaned:
        return "neutral"
    if cleaned in LABELS:
        return cleaned
    return "neutral"


def infer_base_models(sample: dict) -> List[str]:
    models = []
    for key in sample:
        if key.endswith("_1"):
            models.append(key[:-2])
        elif key.endswith("_final"):
            models.append(key[:-6])
    return sorted(set(models))


def collect_drift_rows(records: List[dict], model: str) -> List[dict]:
    rows: List[dict] = []
    for record in records:
        country = str(record.get("Country", "unknown"))
        gold = normalize_label(str(record.get("Gold Label", "")))
        initial = normalize_label(str(record.get(f"{model}_1", "")))
        final = normalize_label(str(record.get(f"{model}_final", "")))
        neutral_collapse = 1 if initial != "neutral" and final == "neutral" else 0
        stance_changed = 1 if initial != final else 0
        accuracy_initial = 1 if initial == gold else 0
        accuracy_final = 1 if final == gold else 0
        rows.append(
            {
                "country": country,
                "gold": gold,
                "initial": initial,
                "final": final,
                "stance_changed": stance_changed,
                "neutral_collapse": neutral_collapse,
                "initial_correct": accuracy_initial,
                "final_correct": accuracy_final,
            }
        )
    return rows


def summarize_rows(rows: List[dict]) -> dict:
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "drift_rate": 0.0,
            "neutral_collapse_rate": 0.0,
            "initial_accuracy": 0.0,
            "final_accuracy": 0.0,
            "accuracy_gain": 0.0,
        }

    drift = sum(row["stance_changed"] for row in rows)
    neutral_collapse = sum(row["neutral_collapse"] for row in rows)
    init_correct = sum(row["initial_correct"] for row in rows)
    final_correct = sum(row["final_correct"] for row in rows)
    return {
        "total": total,
        "drift_rate": drift / total,
        "neutral_collapse_rate": neutral_collapse / total,
        "initial_accuracy": init_correct / total,
        "final_accuracy": final_correct / total,
        "accuracy_gain": (final_correct - init_correct) / total,
    }


def summarize_by_country(rows: List[dict]) -> List[dict]:
    by_country: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_country[row["country"]].append(row)

    output: List[dict] = []
    for country in sorted(by_country):
        country_rows = by_country[country]
        summary = summarize_rows(country_rows)
        output.append(
            {
                "country": country,
                "total": summary["total"],
                "drift_rate": round(summary["drift_rate"], 6),
                "neutral_collapse_rate": round(summary["neutral_collapse_rate"], 6),
                "initial_accuracy": round(summary["initial_accuracy"], 6),
                "final_accuracy": round(summary["final_accuracy"], 6),
                "accuracy_gain": round(summary["accuracy_gain"], 6),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze debate drift from initial to final answers.")
    parser.add_argument("--input_file", required=True, help="Debate JSONL file with *_1 and *_final fields.")
    parser.add_argument("--output_dir", default="results/analysis")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model aliases to analyze.")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    records = list(read_jsonl(input_file))
    if not records:
        raise ValueError(f"Input file has no rows: {input_file}")

    base_models = args.models if args.models else infer_base_models(records[0])
    if not base_models:
        raise ValueError("Could not infer model prefixes. Provide --models explicitly.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_summaries: List[dict] = []
    country_rows_all: List[dict] = []

    for model in base_models:
        drift_rows = collect_drift_rows(records, model)
        summary = summarize_rows(drift_rows)
        model_summaries.append(
            {
                "model": model,
                "total": summary["total"],
                "drift_rate": round(summary["drift_rate"], 6),
                "neutral_collapse_rate": round(summary["neutral_collapse_rate"], 6),
                "initial_accuracy": round(summary["initial_accuracy"], 6),
                "final_accuracy": round(summary["final_accuracy"], 6),
                "accuracy_gain": round(summary["accuracy_gain"], 6),
            }
        )

        country_rows = summarize_by_country(drift_rows)
        for row in country_rows:
            row = dict(row)
            row["model"] = model
            country_rows_all.append(row)

    summary_path = output_dir / "drift_summary.json"
    country_csv = output_dir / "drift_country_metrics.csv"
    model_csv = output_dir / "drift_model_metrics.csv"

    write_json(
        summary_path,
        {
            "input_file": str(input_file),
            "models": model_summaries,
        },
    )
    write_csv(
        model_csv,
        model_summaries,
        ["model", "total", "drift_rate", "neutral_collapse_rate", "initial_accuracy", "final_accuracy", "accuracy_gain"],
    )
    write_csv(
        country_csv,
        country_rows_all,
        ["model", "country", "total", "drift_rate", "neutral_collapse_rate", "initial_accuracy", "final_accuracy", "accuracy_gain"],
    )

    print(f"Saved summary: {summary_path}")
    print(f"Saved model metrics: {model_csv}")
    print(f"Saved country metrics: {country_csv}")


if __name__ == "__main__":
    main()
