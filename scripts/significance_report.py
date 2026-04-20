import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


LABELS = {"yes", "no", "neutral"}


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_label(text: str) -> str:
    cleaned = (text or "").replace(",", "").replace(".", "").strip().lower()
    if "yes" in cleaned or "is socially acceptable" in cleaned or "are socially acceptable" in cleaned:
        return "yes"
    if "no" in cleaned or "is not socially acceptable" in cleaned or "are not socially acceptable" in cleaned:
        return "no"
    if "neither" in cleaned:
        return "neutral"
    if cleaned in LABELS:
        return cleaned
    return "neutral"


def scenario_key(row: dict) -> Tuple[str, str, str, str]:
    return (
        str(row.get("Country", "")),
        str(row.get("Story", "")),
        str(row.get("Rule-of-Thumb", "")),
        str(row.get("Gold Label", "")),
    )


def dedupe_by_scenario(rows: List[dict]) -> List[dict]:
    seen = set()
    out: List[dict] = []
    for row in rows:
        key = scenario_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def compute_metrics(rows: List[dict], model_a: str, model_b: str, parity_lambda: float) -> Dict[str, float]:
    total = len(rows)
    if total == 0:
        return {
            "acc_a": 0.0,
            "acc_b": 0.0,
            "acc_diff_a_minus_b": 0.0,
            "cpg_a": 0.0,
            "cpg_b": 0.0,
            "cpg_diff_a_minus_b": 0.0,
            "paa_a": 0.0,
            "paa_b": 0.0,
            "paa_diff_a_minus_b": 0.0,
        }

    correct_a = 0
    correct_b = 0
    by_country_a: Dict[str, List[int]] = defaultdict(list)
    by_country_b: Dict[str, List[int]] = defaultdict(list)

    field_a = f"{model_a}_final"
    field_b = f"{model_b}_final"

    for row in rows:
        country = str(row.get("Country", "unknown"))
        gold = normalize_label(str(row.get("Gold Label", "")))
        pred_a = normalize_label(str(row.get(field_a, "")))
        pred_b = normalize_label(str(row.get(field_b, "")))

        c_a = int(pred_a == gold)
        c_b = int(pred_b == gold)

        correct_a += c_a
        correct_b += c_b

        by_country_a[country].append(c_a)
        by_country_b[country].append(c_b)

    acc_a = correct_a / total
    acc_b = correct_b / total

    country_acc_a = [sum(v) / len(v) for v in by_country_a.values() if v]
    country_acc_b = [sum(v) / len(v) for v in by_country_b.values() if v]

    cpg_a = (max(country_acc_a) - min(country_acc_a)) if country_acc_a else 0.0
    cpg_b = (max(country_acc_b) - min(country_acc_b)) if country_acc_b else 0.0

    paa_a = acc_a - parity_lambda * cpg_a
    paa_b = acc_b - parity_lambda * cpg_b

    return {
        "acc_a": acc_a,
        "acc_b": acc_b,
        "acc_diff_a_minus_b": acc_a - acc_b,
        "cpg_a": cpg_a,
        "cpg_b": cpg_b,
        "cpg_diff_a_minus_b": cpg_a - cpg_b,
        "paa_a": paa_a,
        "paa_b": paa_b,
        "paa_diff_a_minus_b": paa_a - paa_b,
    }


def percentile_interval(values: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    lo_idx = int((alpha / 2.0) * len(ordered))
    hi_idx = int((1.0 - alpha / 2.0) * len(ordered))
    hi_idx = min(hi_idx, len(ordered) - 1)
    return ordered[lo_idx], ordered[hi_idx]


def bootstrap_diffs(
    rows: List[dict], model_a: str, model_b: str, parity_lambda: float, n_bootstrap: int, seed: int
) -> Dict[str, Tuple[float, float]]:
    rng = random.Random(seed)
    n = len(rows)
    if n == 0:
        return {
            "acc_diff_ci95": (0.0, 0.0),
            "cpg_diff_ci95": (0.0, 0.0),
            "paa_diff_ci95": (0.0, 0.0),
        }

    acc_diffs: List[float] = []
    cpg_diffs: List[float] = []
    paa_diffs: List[float] = []

    for _ in range(n_bootstrap):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        m = compute_metrics(sample, model_a, model_b, parity_lambda)
        acc_diffs.append(m["acc_diff_a_minus_b"])
        cpg_diffs.append(m["cpg_diff_a_minus_b"])
        paa_diffs.append(m["paa_diff_a_minus_b"])

    return {
        "acc_diff_ci95": percentile_interval(acc_diffs),
        "cpg_diff_ci95": percentile_interval(cpg_diffs),
        "paa_diff_ci95": percentile_interval(paa_diffs),
    }


def permutation_pvalues(
    rows: List[dict], model_a: str, model_b: str, parity_lambda: float, n_perm: int, seed: int
) -> Dict[str, float]:
    rng = random.Random(seed)
    if not rows:
        return {"acc_diff_pvalue": 1.0, "cpg_diff_pvalue": 1.0, "paa_diff_pvalue": 1.0}

    observed = compute_metrics(rows, model_a, model_b, parity_lambda)
    obs_acc = observed["acc_diff_a_minus_b"]
    obs_cpg = observed["cpg_diff_a_minus_b"]
    obs_paa = observed["paa_diff_a_minus_b"]

    field_a = f"{model_a}_final"
    field_b = f"{model_b}_final"

    ge_acc = 0
    ge_cpg = 0
    ge_paa = 0

    for _ in range(n_perm):
        shuffled_rows: List[dict] = []
        for row in rows:
            new_row = dict(row)
            if rng.random() < 0.5:
                new_row[field_a], new_row[field_b] = row.get(field_b, ""), row.get(field_a, "")
            shuffled_rows.append(new_row)

        m = compute_metrics(shuffled_rows, model_a, model_b, parity_lambda)
        if abs(m["acc_diff_a_minus_b"]) >= abs(obs_acc):
            ge_acc += 1
        if abs(m["cpg_diff_a_minus_b"]) >= abs(obs_cpg):
            ge_cpg += 1
        if abs(m["paa_diff_a_minus_b"]) >= abs(obs_paa):
            ge_paa += 1

    # Add-one smoothing.
    return {
        "acc_diff_pvalue": (ge_acc + 1) / (n_perm + 1),
        "cpg_diff_pvalue": (ge_cpg + 1) / (n_perm + 1),
        "paa_diff_pvalue": (ge_paa + 1) / (n_perm + 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired significance report for two debaters in one debate output file.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--first_model", required=True)
    parser.add_argument("--second_model", required=True)
    parser.add_argument("--parity_lambda", type=float, default=0.25)
    parser.add_argument("--bootstrap_samples", type=int, default=1500)
    parser.add_argument("--permutation_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_json = (
        Path(args.output_json)
        if args.output_json
        else input_file.with_name(input_file.stem + "_significance_report.json")
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows = dedupe_by_scenario(read_jsonl(input_file))

    observed = compute_metrics(rows, args.first_model, args.second_model, args.parity_lambda)
    bootstrap = bootstrap_diffs(
        rows,
        args.first_model,
        args.second_model,
        args.parity_lambda,
        args.bootstrap_samples,
        args.seed,
    )
    pvals = permutation_pvalues(
        rows,
        args.first_model,
        args.second_model,
        args.parity_lambda,
        args.permutation_samples,
        args.seed + 1,
    )

    payload = {
        "input_file": str(input_file),
        "n_unique_rows": len(rows),
        "first_model": args.first_model,
        "second_model": args.second_model,
        "parity_lambda": args.parity_lambda,
        "observed": {k: round(v, 6) for k, v in observed.items()},
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_ci95": {
            k: [round(v[0], 6), round(v[1], 6)] for k, v in bootstrap.items()
        },
        "permutation_samples": args.permutation_samples,
        "pvalues_two_sided": {k: round(v, 6) for k, v in pvals.items()},
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved significance report: {output_json}")
    print(f"n_unique_rows: {len(rows)}")
    print(f"acc_diff ({args.first_model}-{args.second_model}): {payload['observed']['acc_diff_a_minus_b']:.4f}")


if __name__ == "__main__":
    main()
