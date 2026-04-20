import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_country_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "country",
        "total",
        "correct",
        "accuracy",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_label(text: str) -> str:
    cleaned = (text or "").replace(",", "").replace(".", "").strip().lower()
    if "yes" in cleaned or "is socially acceptable" in cleaned or "are socially acceptable" in cleaned:
        return "yes"
    if "no" in cleaned or "is not socially acceptable" in cleaned or "are not socially acceptable" in cleaned:
        return "no"
    if "neither" in cleaned:
        return "neutral"
    return cleaned


def infer_prediction_field(sample: dict, mode: str, model: Optional[str], prediction_field: Optional[str]) -> str:
    if prediction_field:
        return prediction_field
    if model:
        if mode == "single":
            if f"{model}_final" in sample:
                return f"{model}_final"
            return model
        if mode == "multi":
            return f"{model}_final"

    candidate_fields = [
        key
        for key in sample
        if key not in {"Country", "Story", "Rule-of-Thumb", "Gold Label"}
    ]

    # Prefer *_final fields if available.
    finals = [k for k in candidate_fields if k.endswith("_final")]
    if finals:
        return finals[0]
    if candidate_fields:
        return candidate_fields[0]

    raise ValueError("Could not infer prediction field from input file. Use --prediction_field.")


@dataclass
class CountryStats:
    total: int = 0
    correct: int = 0


@dataclass
class Metrics:
    global_accuracy: float
    macro_accuracy: float
    best_country_accuracy: float
    worst_country_accuracy: float
    cpg: float
    paa: float


def compute_country_accuracies(country_stats: Dict[str, CountryStats]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for country, stats in country_stats.items():
        if stats.total == 0:
            out[country] = 0.0
        else:
            out[country] = stats.correct / stats.total
    return out


def compute_metrics(country_stats: Dict[str, CountryStats], total: int, correct: int, parity_lambda: float) -> Metrics:
    country_acc = compute_country_accuracies(country_stats)
    if not country_acc:
        raise ValueError("No country-level data found.")

    global_acc = (correct / total) if total else 0.0
    macro_acc = sum(country_acc.values()) / len(country_acc)
    best = max(country_acc.values())
    worst = min(country_acc.values())
    cpg = best - worst
    paa = global_acc - parity_lambda * cpg
    return Metrics(
        global_accuracy=global_acc,
        macro_accuracy=macro_acc,
        best_country_accuracy=best,
        worst_country_accuracy=worst,
        cpg=cpg,
        paa=paa,
    )


def bootstrap_ci(
    rows: List[Tuple[str, str, str]],
    parity_lambda: float,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    random.seed(seed)
    n = len(rows)
    if n == 0:
        return {}

    global_values: List[float] = []
    macro_values: List[float] = []
    cpg_values: List[float] = []
    paa_values: List[float] = []

    for _ in range(bootstrap_samples):
        sampled = [rows[random.randrange(n)] for _ in range(n)]
        country_stats: Dict[str, CountryStats] = {}
        total = 0
        correct = 0

        for country, gold, pred in sampled:
            if country not in country_stats:
                country_stats[country] = CountryStats()
            country_stats[country].total += 1
            total += 1
            if pred == gold:
                country_stats[country].correct += 1
                correct += 1

        m = compute_metrics(country_stats, total, correct, parity_lambda)
        global_values.append(m.global_accuracy)
        macro_values.append(m.macro_accuracy)
        cpg_values.append(m.cpg)
        paa_values.append(m.paa)

    def interval(values: List[float]) -> Tuple[float, float]:
        values = sorted(values)
        lo_idx = int(0.025 * len(values))
        hi_idx = int(0.975 * len(values))
        hi_idx = min(hi_idx, len(values) - 1)
        return values[lo_idx], values[hi_idx]

    return {
        "global_accuracy_ci95": interval(global_values),
        "macro_accuracy_ci95": interval(macro_values),
        "cpg_ci95": interval(cpg_values),
        "paa_ci95": interval(paa_values),
    }


def run_one(
    input_file: Path,
    mode: str,
    model: Optional[str],
    prediction_field: Optional[str],
    parity_lambda: float,
    bootstrap_samples: int,
    seed: int,
) -> Tuple[dict, List[dict]]:
    records = list(read_jsonl(input_file))
    if not records:
        raise ValueError(f"Input file has no rows: {input_file}")

    pred_field = infer_prediction_field(records[0], mode, model, prediction_field)

    country_stats: Dict[str, CountryStats] = {}
    rows_for_bootstrap: List[Tuple[str, str, str]] = []
    total = 0
    correct = 0

    for row in records:
        country = str(row.get("Country", "unknown"))
        gold = normalize_label(str(row.get("Gold Label", "")))
        pred = normalize_label(str(row.get(pred_field, "")))

        if country not in country_stats:
            country_stats[country] = CountryStats()
        country_stats[country].total += 1
        total += 1

        if pred == gold:
            country_stats[country].correct += 1
            correct += 1

        rows_for_bootstrap.append((country, gold, pred))

    metrics = compute_metrics(country_stats, total, correct, parity_lambda)
    ci = bootstrap_ci(rows_for_bootstrap, parity_lambda, bootstrap_samples, seed)

    per_country_rows: List[dict] = []
    for country in sorted(country_stats):
        stats = country_stats[country]
        acc = (stats.correct / stats.total) if stats.total else 0.0
        per_country_rows.append(
            {
                "country": country,
                "total": stats.total,
                "correct": stats.correct,
                "accuracy": round(acc, 6),
            }
        )

    summary = {
        "input_file": str(input_file),
        "mode": mode,
        "prediction_field": pred_field,
        "total": total,
        "correct": correct,
        "parity_lambda": parity_lambda,
        "global_accuracy": round(metrics.global_accuracy, 6),
        "macro_accuracy": round(metrics.macro_accuracy, 6),
        "best_country_accuracy": round(metrics.best_country_accuracy, 6),
        "worst_country_accuracy": round(metrics.worst_country_accuracy, 6),
        "cpg": round(metrics.cpg, 6),
        "paa": round(metrics.paa, 6),
        "bootstrap_samples": bootstrap_samples,
        "ci95": {
            key: [round(lo, 6), round(hi, 6)]
            for key, (lo, hi) in ci.items()
        },
    }
    return summary, per_country_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Cultural benchmark metrics (CPG/PAA) for one output file.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--mode", choices=["single", "multi", "auto"], default="auto")
    parser.add_argument("--model", default="", help="Model name for field inference (e.g., gemma)")
    parser.add_argument("--prediction_field", default="", help="Explicit prediction field in JSONL")
    parser.add_argument("--parity_lambda", type=float, default=0.25)
    parser.add_argument("--bootstrap_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_country_csv", default="")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    mode = args.mode
    if mode == "auto":
        mode = "single"

    output_json = Path(args.output_json) if args.output_json else input_file.with_name(input_file.stem + "_benchmark.json")
    output_country_csv = (
        Path(args.output_country_csv)
        if args.output_country_csv
        else input_file.with_name(input_file.stem + "_country_metrics.csv")
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_country_csv.parent.mkdir(parents=True, exist_ok=True)

    model = args.model if args.model else None
    prediction_field = args.prediction_field if args.prediction_field else None

    summary, per_country = run_one(
        input_file=input_file,
        mode=mode,
        model=model,
        prediction_field=prediction_field,
        parity_lambda=args.parity_lambda,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    write_json(output_json, summary)
    write_country_csv(output_country_csv, per_country)

    print("Saved summary:", output_json)
    print("Saved country metrics:", output_country_csv)
    print("Global accuracy:", f"{summary['global_accuracy']:.4f}")
    print("CPG:", f"{summary['cpg']:.4f}")
    print("PAA:", f"{summary['paa']:.4f}")


if __name__ == "__main__":
    main()
