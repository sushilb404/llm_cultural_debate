import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from label_utils import normalize_label


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


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class CountryStats:
    total: int = 0
    correct: int = 0


def compute_metrics_from_rows(rows: List[Tuple[str, str, str]], parity_lambda: float) -> dict:
    country: Dict[str, CountryStats] = {}
    total = 0
    correct = 0

    for c, gold, pred in rows:
        if c not in country:
            country[c] = CountryStats()
        country[c].total += 1
        total += 1
        if pred == gold:
            country[c].correct += 1
            correct += 1

    country_acc = {k: (v.correct / v.total if v.total else 0.0) for k, v in country.items()}
    global_acc = (correct / total) if total else 0.0
    macro_acc = (sum(country_acc.values()) / len(country_acc)) if country_acc else 0.0
    best = max(country_acc.values()) if country_acc else 0.0
    worst = min(country_acc.values()) if country_acc else 0.0
    cpg = best - worst
    paa = global_acc - parity_lambda * cpg

    return {
        "global_accuracy": global_acc,
        "macro_accuracy": macro_acc,
        "best_country_accuracy": best,
        "worst_country_accuracy": worst,
        "cpg": cpg,
        "paa": paa,
        "country_accuracy": country_acc,
    }


def parse_single_model_from_file(path: Path) -> Optional[str]:
    stem = path.stem
    suffixes = ["_without_rot", "_with_rot", "_self_reflection"]
    for s in suffixes:
        if stem.endswith(s):
            return stem[: -len(s)]
    # If stem has no suffix, assume model name directly.
    return stem if stem else None


def parse_pair_from_file(path: Path) -> Optional[Tuple[str, str]]:
    stem = path.stem
    if "_" not in stem:
        return None
    left, right = stem.split("_", 1)
    if not left or not right:
        return None
    return left, right


def collect_single_metrics(single_dir: Path, parity_lambda: float) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for file in sorted(single_dir.glob("*.jsonl")):
        model = parse_single_model_from_file(file)
        if not model:
            continue

        rows: List[Tuple[str, str, str]] = []
        for record in read_jsonl(file):
            country = str(record.get("Country", "unknown"))
            gold = normalize_label(str(record.get("Gold Label", "")))

            field = model
            if field not in record and f"{model}_final" in record:
                field = f"{model}_final"
            pred = normalize_label(str(record.get(field, "")))
            rows.append((country, gold, pred))

        if rows:
            out[model] = compute_metrics_from_rows(rows, parity_lambda)
            out[model]["source_file"] = str(file)
    return out


def collect_multi_metrics(multi_dir: Path, parity_lambda: float) -> List[dict]:
    pair_rows: List[dict] = []
    for file in sorted(multi_dir.glob("*.jsonl")):
        pair = parse_pair_from_file(file)
        if not pair:
            continue
        first, second = pair

        rows_first: List[Tuple[str, str, str]] = []
        rows_second: List[Tuple[str, str, str]] = []

        for record in read_jsonl(file):
            country = str(record.get("Country", "unknown"))
            gold = normalize_label(str(record.get("Gold Label", "")))
            pred_first = normalize_label(str(record.get(f"{first}_final", "")))
            pred_second = normalize_label(str(record.get(f"{second}_final", "")))
            rows_first.append((country, gold, pred_first))
            rows_second.append((country, gold, pred_second))

        if not rows_first:
            continue

        metrics_first = compute_metrics_from_rows(rows_first, parity_lambda)
        metrics_second = compute_metrics_from_rows(rows_second, parity_lambda)
        pair_rows.append(
            {
                "file": str(file),
                "pair": f"{first}_{second}",
                "first": first,
                "second": second,
                "metrics_first": metrics_first,
                "metrics_second": metrics_second,
            }
        )
    return pair_rows


def variance(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Intervention analysis: DPG, PSI, and partner selection.")
    parser.add_argument("--single_dir", required=True)
    parser.add_argument("--multi_dir", required=True)
    parser.add_argument("--output_dir", default="results/analysis")
    parser.add_argument("--parity_lambda", type=float, default=0.25)
    args = parser.parse_args()

    single_dir = Path(args.single_dir)
    multi_dir = Path(args.multi_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not single_dir.exists():
        raise FileNotFoundError(f"single_dir not found: {single_dir}")
    if not multi_dir.exists():
        raise FileNotFoundError(f"multi_dir not found: {multi_dir}")

    single_metrics = collect_single_metrics(single_dir, args.parity_lambda)
    multi_metrics = collect_multi_metrics(multi_dir, args.parity_lambda)

    if not single_metrics:
        raise ValueError("No single-model metrics found. Ensure single_dir contains output .jsonl files.")
    if not multi_metrics:
        raise ValueError("No multi-model metrics found. Ensure multi_dir contains debate output .jsonl files.")

    # Base model -> partner rows containing debate metrics from that model's perspective.
    by_base: Dict[str, List[dict]] = defaultdict(list)

    for pair in multi_metrics:
        first = pair["first"]
        second = pair["second"]
        by_base[first].append({"partner": second, "metrics": pair["metrics_first"], "pair": pair["pair"]})
        by_base[second].append({"partner": first, "metrics": pair["metrics_second"], "pair": pair["pair"]})

    dpg_rows: List[dict] = []
    psi_rows: List[dict] = []
    selected_partner_rows: List[dict] = []

    for base_model, partner_rows in sorted(by_base.items()):
        if base_model not in single_metrics:
            continue

        single = single_metrics[base_model]
        country_acc_by_partner: Dict[str, Dict[str, float]] = {}

        for row in partner_rows:
            partner = row["partner"]
            m = row["metrics"]
            country_acc_by_partner[partner] = m["country_accuracy"]

            dpg_rows.append(
                {
                    "base_model": base_model,
                    "partner_model": partner,
                    "pair": row["pair"],
                    "single_global_accuracy": round(single["global_accuracy"], 6),
                    "debate_global_accuracy": round(m["global_accuracy"], 6),
                    "delta_global_accuracy": round(m["global_accuracy"] - single["global_accuracy"], 6),
                    "single_cpg": round(single["cpg"], 6),
                    "debate_cpg": round(m["cpg"], 6),
                    "dpg": round(single["cpg"] - m["cpg"], 6),
                    "single_paa": round(single["paa"], 6),
                    "debate_paa": round(m["paa"], 6),
                    "delta_paa": round(m["paa"] - single["paa"], 6),
                }
            )

        # Partner Stability Index: mean variance of per-country accuracy across partners.
        all_countries = sorted({c for stats in country_acc_by_partner.values() for c in stats})
        country_variances: List[float] = []
        for c in all_countries:
            values = [country_acc_by_partner[p].get(c, 0.0) for p in country_acc_by_partner]
            country_variances.append(variance(values))

        psi = (sum(country_variances) / len(country_variances)) if country_variances else 0.0
        psi_rows.append(
            {
                "base_model": base_model,
                "partners_evaluated": len(partner_rows),
                "psi": round(psi, 6),
            }
        )

        # Partner selection by PAA.
        best = max(partner_rows, key=lambda r: r["metrics"]["paa"])
        selected_partner_rows.append(
            {
                "base_model": base_model,
                "selected_partner": best["partner"],
                "selected_pair": best["pair"],
                "selected_paa": round(best["metrics"]["paa"], 6),
                "single_paa": round(single["paa"], 6),
                "paa_gain": round(best["metrics"]["paa"] - single["paa"], 6),
                "selected_global_accuracy": round(best["metrics"]["global_accuracy"], 6),
                "single_global_accuracy": round(single["global_accuracy"], 6),
            }
        )

    dpg_csv = output_dir / "debate_parity_gain.csv"
    psi_csv = output_dir / "partner_stability_index.csv"
    selection_csv = output_dir / "partner_selection_by_paa.csv"
    summary_json = output_dir / "intervention_summary.json"

    write_csv(
        dpg_csv,
        dpg_rows,
        [
            "base_model",
            "partner_model",
            "pair",
            "single_global_accuracy",
            "debate_global_accuracy",
            "delta_global_accuracy",
            "single_cpg",
            "debate_cpg",
            "dpg",
            "single_paa",
            "debate_paa",
            "delta_paa",
        ],
    )
    write_csv(psi_csv, psi_rows, ["base_model", "partners_evaluated", "psi"])
    write_csv(
        selection_csv,
        selected_partner_rows,
        [
            "base_model",
            "selected_partner",
            "selected_pair",
            "selected_paa",
            "single_paa",
            "paa_gain",
            "selected_global_accuracy",
            "single_global_accuracy",
        ],
    )

    summary = {
        "parity_lambda": args.parity_lambda,
        "single_models": sorted(single_metrics.keys()),
        "num_multi_pairs": len(multi_metrics),
        "outputs": {
            "dpg_csv": str(dpg_csv),
            "psi_csv": str(psi_csv),
            "selection_csv": str(selection_csv),
        },
    }
    write_json(summary_json, summary)

    print("Saved:", dpg_csv)
    print("Saved:", psi_csv)
    print("Saved:", selection_csv)
    print("Saved:", summary_json)


if __name__ == "__main__":
    main()
