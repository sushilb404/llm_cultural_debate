import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_benchmarks(analysis_dir: Path) -> List[dict]:
    benchmarks = []
    for path in sorted(analysis_dir.glob("*_benchmark.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        model_name = path.stem.replace("_benchmark", "")
        data["model"] = model_name
        benchmarks.append(data)
    return benchmarks


def load_country_tables(analysis_dir: Path) -> Dict[str, List[dict]]:
    tables: Dict[str, List[dict]] = {}
    for path in sorted(analysis_dir.glob("*_country_metrics.csv")):
        model_name = path.stem.replace("_country_metrics", "")
        rows = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "correct" not in reader.fieldnames:
                continue
            for row in reader:
                rows.append(
                    {
                        "country": row["country"],
                        "total": int(row["total"]),
                        "correct": int(row["correct"]),
                        "accuracy": float(row["accuracy"]),
                    }
                )
        tables[model_name] = rows
    return tables


def load_intervention_rows(analysis_dir: Path) -> List[dict]:
    path = analysis_dir / "debate_parity_gain.csv"
    rows: List[dict] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "base_model": row["base_model"],
                    "partner_model": row["partner_model"],
                    "pair": row["pair"],
                    "single_global_accuracy": float(row["single_global_accuracy"]),
                    "debate_global_accuracy": float(row["debate_global_accuracy"]),
                    "delta_global_accuracy": float(row["delta_global_accuracy"]),
                    "single_cpg": float(row["single_cpg"]),
                    "debate_cpg": float(row["debate_cpg"]),
                    "dpg": float(row["dpg"]),
                    "single_paa": float(row["single_paa"]),
                    "debate_paa": float(row["debate_paa"]),
                    "delta_paa": float(row["delta_paa"]),
                }
            )

    # Deduplicate repeated pair records.
    deduped = {}
    for row in rows:
        key = (row["base_model"], row["partner_model"])
        if key not in deduped:
            deduped[key] = row
    return list(deduped.values())


def load_psi_rows(analysis_dir: Path) -> List[dict]:
    path = analysis_dir / "partner_stability_index.csv"
    rows: List[dict] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "base_model": row["base_model"],
                    "partners_evaluated": int(row["partners_evaluated"]),
                    "psi": float(row["psi"]),
                }
            )
    return rows


def load_drift_rows(analysis_dir: Path) -> List[dict]:
    path = analysis_dir / "drift_model_metrics.csv"
    rows: List[dict] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "model": row["model"],
                    "total": int(row["total"]),
                    "drift_rate": float(row["drift_rate"]),
                    "neutral_collapse_rate": float(row["neutral_collapse_rate"]),
                    "initial_accuracy": float(row["initial_accuracy"]),
                    "final_accuracy": float(row["final_accuracy"]),
                    "accuracy_gain": float(row["accuracy_gain"]),
                }
            )
    return rows


def load_drift_country_tables(analysis_dir: Path) -> Dict[str, List[dict]]:
    tables: Dict[str, List[dict]] = {}
    path = analysis_dir / "drift_country_metrics.csv"
    if not path.exists():
        return tables

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            tables.setdefault(model, []).append(
                {
                    "country": row["country"],
                    "total": int(row["total"]),
                    "drift_rate": float(row["drift_rate"]),
                    "neutral_collapse_rate": float(row["neutral_collapse_rate"]),
                    "initial_accuracy": float(row["initial_accuracy"]),
                    "final_accuracy": float(row["final_accuracy"]),
                    "accuracy_gain": float(row["accuracy_gain"]),
                }
            )
    return tables


def save_model_overview(benchmarks: List[dict], output_dir: Path) -> None:
    models = [b["model"] for b in benchmarks]
    acc = [b["global_accuracy"] for b in benchmarks]
    paa = [b["paa"] for b in benchmarks]

    x = list(range(len(models)))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([v - width / 2 for v in x], acc, width=width, label="Global Accuracy")
    plt.bar([v + width / 2 for v in x], paa, width=width, label="PAA")
    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Accuracy vs PAA by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_overview_accuracy_paa.png", dpi=180)
    plt.close()


def save_fairness_scatter(benchmarks: List[dict], output_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    for b in benchmarks:
        plt.scatter(b["cpg"], b["global_accuracy"], s=100)
        plt.annotate(b["model"], (b["cpg"], b["global_accuracy"]), xytext=(5, 5), textcoords="offset points")

    plt.xlabel("CPG (Lower is Better)")
    plt.ylabel("Global Accuracy")
    plt.title("Fairness-Performance Tradeoff")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "fairness_performance_scatter.png", dpi=180)
    plt.close()


def save_country_heatmap(country_tables: Dict[str, List[dict]], output_dir: Path) -> None:
    if not country_tables:
        return

    models = sorted(country_tables.keys())
    all_countries = sorted({row["country"] for rows in country_tables.values() for row in rows})

    matrix: List[List[float]] = []
    for country in all_countries:
        row_vals = []
        for model in models:
            lookup = {r["country"]: r["accuracy"] for r in country_tables[model]}
            row_vals.append(lookup.get(country, 0.0))
        matrix.append(row_vals)

    plt.figure(figsize=(max(6, len(models) * 1.5), max(5, len(all_countries) * 0.45)))
    image = plt.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    plt.colorbar(image, label="Accuracy")
    plt.xticks(range(len(models)), models)
    plt.yticks(range(len(all_countries)), all_countries)
    plt.title("Per-Country Accuracy Heatmap")

    for i in range(len(all_countries)):
        for j in range(len(models)):
            plt.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "country_accuracy_heatmap.png", dpi=180)
    plt.close()


def save_country_coverage(country_tables: Dict[str, List[dict]], output_dir: Path) -> None:
    if not country_tables:
        return

    models = sorted(country_tables.keys())
    totals = []
    for model in models:
        totals.append(sum(r["total"] for r in country_tables[model]))

    plt.figure(figsize=(7, 4.5))
    plt.bar(models, totals)
    plt.ylabel("Examples")
    plt.title("Sample Count by Model Output")
    plt.tight_layout()
    plt.savefig(output_dir / "sample_count_by_model.png", dpi=180)
    plt.close()


def aggregate_by_base_model(rows: List[dict], value_key: str) -> Tuple[List[str], List[float]]:
    by_model: Dict[str, List[float]] = {}
    for row in rows:
        by_model.setdefault(row["base_model"], []).append(row[value_key])
    models = sorted(by_model.keys())
    values = [sum(by_model[m]) / len(by_model[m]) for m in models]
    return models, values


def save_intervention_bars(intervention_rows: List[dict], output_dir: Path) -> None:
    if not intervention_rows:
        return

    models, delta_paa = aggregate_by_base_model(intervention_rows, "delta_paa")
    _, dpg_vals = aggregate_by_base_model(intervention_rows, "dpg")

    plt.figure(figsize=(8, 4.5))
    plt.bar(models, delta_paa)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Delta PAA")
    plt.title("Intervention Effect on PAA")
    plt.tight_layout()
    plt.savefig(output_dir / "intervention_delta_paa.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(models, dpg_vals)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("DPG")
    plt.title("Debate Parity Gain by Base Model")
    plt.tight_layout()
    plt.savefig(output_dir / "intervention_dpg.png", dpi=180)
    plt.close()


def save_psi(psi_rows: List[dict], output_dir: Path) -> None:
    if not psi_rows:
        return

    models = [r["base_model"] for r in sorted(psi_rows, key=lambda x: x["base_model"])]
    psi_vals = [r["psi"] for r in sorted(psi_rows, key=lambda x: x["base_model"])]

    plt.figure(figsize=(7, 4.5))
    plt.bar(models, psi_vals)
    plt.ylabel("PSI")
    plt.title("Partner Stability Index")
    plt.tight_layout()
    plt.savefig(output_dir / "partner_stability_index.png", dpi=180)
    plt.close()


def save_drift_overview(drift_rows: List[dict], output_dir: Path) -> None:
    if not drift_rows:
        return

    models = [r["model"] for r in drift_rows]
    drift = [r["drift_rate"] for r in drift_rows]
    neutral = [r["neutral_collapse_rate"] for r in drift_rows]
    initial = [r["initial_accuracy"] for r in drift_rows]
    final = [r["final_accuracy"] for r in drift_rows]

    x = list(range(len(models)))
    width = 0.22

    plt.figure(figsize=(8.5, 4.8))
    plt.bar([v - width for v in x], drift, width=width, label="Drift Rate")
    plt.bar(x, neutral, width=width, label="Neutral Collapse")
    plt.bar([v + width for v in x], [f - i for f, i in zip(final, initial)], width=width, label="Accuracy Gain")
    plt.xticks(x, models)
    plt.ylim(-0.2, 1.0)
    plt.ylabel("Rate / Gain")
    plt.title("Debate Drift and Stability Summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "drift_overview.png", dpi=180)
    plt.close()


def save_drift_gap(drift_rows: List[dict], output_dir: Path) -> None:
    if not drift_rows:
        return

    models = [r["model"] for r in drift_rows]
    gaps = [r["final_accuracy"] - r["initial_accuracy"] for r in drift_rows]

    plt.figure(figsize=(7, 4.5))
    plt.bar(models, gaps)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Final - Initial Accuracy")
    plt.title("Accuracy Change Through Debate")
    plt.tight_layout()
    plt.savefig(output_dir / "drift_accuracy_gain.png", dpi=180)
    plt.close()


def save_drift_country_heatmap(drift_country_tables: Dict[str, List[dict]], output_dir: Path) -> None:
    if not drift_country_tables:
        return

    models = sorted(drift_country_tables.keys())
    countries = sorted({row["country"] for rows in drift_country_tables.values() for row in rows})

    matrix: List[List[float]] = []
    for country in countries:
        row_vals = []
        for model in models:
            lookup = {r["country"]: r["drift_rate"] for r in drift_country_tables[model]}
            row_vals.append(lookup.get(country, 0.0))
        matrix.append(row_vals)

    plt.figure(figsize=(max(6, len(models) * 1.5), max(5, len(countries) * 0.45)))
    image = plt.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=1)
    plt.colorbar(image, label="Drift Rate")
    plt.xticks(range(len(models)), models)
    plt.yticks(range(len(countries)), countries)
    plt.title("Per-Country Drift Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "drift_country_heatmap.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark and intervention plots from results/analysis.")
    parser.add_argument("--analysis_dir", default="results/analysis")
    parser.add_argument("--output_dir", default="results/figures")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = load_benchmarks(analysis_dir)
    country_tables = load_country_tables(analysis_dir)
    drift_country_tables = load_drift_country_tables(analysis_dir)
    intervention_rows = load_intervention_rows(analysis_dir)
    psi_rows = load_psi_rows(analysis_dir)
    drift_rows = load_drift_rows(analysis_dir)

    if benchmarks:
        save_model_overview(benchmarks, output_dir)
        save_fairness_scatter(benchmarks, output_dir)
    if country_tables:
        save_country_heatmap(country_tables, output_dir)
        save_country_coverage(country_tables, output_dir)
    if intervention_rows:
        save_intervention_bars(intervention_rows, output_dir)
    if psi_rows:
        save_psi(psi_rows, output_dir)
    if drift_rows:
        save_drift_overview(drift_rows, output_dir)
        save_drift_gap(drift_rows, output_dir)
    if drift_country_tables:
        save_drift_country_heatmap(drift_country_tables, output_dir)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
