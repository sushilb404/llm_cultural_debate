from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from label_utils import normalize_label


LABELS = ("yes", "no", "neutral")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def word_count(text: object) -> int:
    return len(str(text or "").split())


def scenario_id(row: dict) -> str:
    key = "\n".join(
        [
            str(row.get("Country", "")),
            str(row.get("Story", "")),
            str(row.get("Rule-of-Thumb", "")),
            str(row.get("Gold Label", "")),
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def parse_spec(spec: str) -> tuple[str, Path, list[str]]:
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise ValueError(f"Run spec must be name=path=model1,model2: {spec}")
    name, path_text, model_text = parts
    models = [m.strip() for m in model_text.split(",") if m.strip()]
    if not models:
        raise ValueError(f"Run spec has no models: {spec}")
    return name.strip(), Path(path_text), models


def candidate_field(row: dict, model: str) -> str | None:
    for field in (f"{model}_final", model):
        if field in row:
            return field
    return None


def collect_long_records(specs: Iterable[str]) -> list[dict]:
    records: list[dict] = []
    for spec in specs:
        run_name, rel_path, models = parse_spec(spec)
        path = rel_path if rel_path.is_absolute() else REPO_ROOT / rel_path
        rows = read_jsonl(path)
        if not rows:
            continue
        for row_index, row in enumerate(rows):
            gold = normalize_label(str(row.get("Gold Label", "")))
            sid = scenario_id(row)
            for model in models:
                field = candidate_field(row, model)
                if field is None:
                    continue
                pred = normalize_label(str(row.get(field, "")))
                records.append(
                    {
                        "run": run_name,
                        "source_file": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
                        "row_index": row_index,
                        "scenario_id": sid,
                        "country": str(row.get("Country", "")),
                        "gold": gold,
                        "model": model,
                        "field": field,
                        "pred": pred,
                        "correct": int(pred == gold),
                        "valid_pred": int(pred in LABELS),
                        "story_words": word_count(row.get("Story", "")),
                        "rot_words": word_count(row.get("Rule-of-Thumb", "")),
                        "has_debate_turns": int(f"{model}_1" in row or f"{model}_2" in row),
                    }
                )
    return records


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def model_metrics(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["run"], record["model"])].append(record)

    rows = []
    for (run_name, model), items in sorted(groups.items()):
        pred_counts = Counter(item["pred"] for item in items)
        rows.append(
            {
                "run": run_name,
                "model": model,
                "n": len(items),
                "accuracy": sum(item["correct"] for item in items) / len(items),
                "valid_rate": sum(item["valid_pred"] for item in items) / len(items),
                "yes_rate": pred_counts.get("yes", 0) / len(items),
                "no_rate": pred_counts.get("no", 0) / len(items),
                "neutral_rate": pred_counts.get("neutral", 0) / len(items),
                "invalid_rate": pred_counts.get("invalid", 0) / len(items),
            }
        )
    return rows


def pairwise_agreement(records: list[dict]) -> list[dict]:
    by_scenario: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for record in records:
        by_scenario[(record["run"], record["scenario_id"])][record["model"]] = record

    pairs: dict[tuple[str, str, str], list[tuple[dict, dict]]] = defaultdict(list)
    for (run_name, _sid), model_records in by_scenario.items():
        models = sorted(model_records)
        for i, left in enumerate(models):
            for right in models[i + 1 :]:
                pairs[(run_name, left, right)].append((model_records[left], model_records[right]))

    rows = []
    for (run_name, left, right), items in sorted(pairs.items()):
        agree = sum(1 for lrow, rrow in items if lrow["pred"] == rrow["pred"])
        both_correct = sum(1 for lrow, rrow in items if lrow["correct"] and rrow["correct"])
        left_only = sum(1 for lrow, rrow in items if lrow["correct"] and not rrow["correct"])
        right_only = sum(1 for lrow, rrow in items if rrow["correct"] and not lrow["correct"])
        rows.append(
            {
                "run": run_name,
                "left_model": left,
                "right_model": right,
                "n": len(items),
                "agreement": agree / len(items),
                "both_correct_rate": both_correct / len(items),
                "left_only_correct_rate": left_only / len(items),
                "right_only_correct_rate": right_only / len(items),
            }
        )
    return rows


def run_random_forest(records: list[dict], output_dir: Path, seed: int) -> dict:
    if len(records) < 20 or len({r["correct"] for r in records}) < 2:
        return {"status": "skipped", "reason": "not enough labeled variation"}

    feature_dicts = []
    y = []
    for record in records:
        feature_dicts.append(
            {
                "run": record["run"],
                "model": record["model"],
                "country": record["country"],
                "pred": record["pred"],
                "gold": record["gold"],
                "story_words": record["story_words"],
                "rot_words": record["rot_words"],
                "has_debate_turns": record["has_debate_turns"],
            }
        )
        y.append(record["correct"])

    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(feature_dicts)
    y_array = np.asarray(y, dtype=int)

    stratify = y_array if min(Counter(y_array).values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_array, test_size=0.25, random_state=seed, stratify=stratify
    )
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        class_weight="balanced",
        min_samples_leaf=3,
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    importance_rows = [
        {"feature": name, "importance": float(value)}
        for name, value in sorted(
            zip(vectorizer.get_feature_names_out(), rf.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    write_csv(output_dir / "rf_feature_importance.csv", importance_rows, ["feature", "importance"])

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    report["heldout_accuracy"] = accuracy_score(y_test, pred)
    report["n_train"] = int(len(y_train))
    report["n_test"] = int(len(y_test))
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return {
        "status": "ok",
        "heldout_accuracy": float(report["heldout_accuracy"]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def scenario_clusters(records: list[dict], output_dir: Path, k: int, seed: int) -> dict:
    models = sorted({record["model"] for record in records})
    scenario_meta: dict[str, dict] = {}
    correctness: dict[str, dict[str, int]] = defaultdict(dict)
    for record in records:
        sid = record["scenario_id"]
        scenario_meta.setdefault(
            sid,
            {
                "scenario_id": sid,
                "country": record["country"],
                "gold": record["gold"],
                "story_words": record["story_words"],
                "rot_words": record["rot_words"],
            },
        )
        correctness[sid][f"{record['run']}::{record['model']}"] = record["correct"]

    columns = sorted({key for item in correctness.values() for key in item})
    if not columns:
        return {"status": "skipped", "reason": "no scenario columns"}

    rows = []
    X = []
    for sid, values in sorted(correctness.items()):
        vector = [values.get(col, -1) for col in columns]
        X.append(vector)
        row = dict(scenario_meta[sid])
        for col, value in zip(columns, vector):
            row[col] = value
        rows.append(row)

    X_array = np.asarray(X, dtype=float)
    n_clusters = min(max(k, 1), len(rows))
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(X_array)
    for row, label in zip(rows, labels):
        row["cluster"] = int(label)
        valid = [row[col] for col in columns if row[col] >= 0]
        row["mean_correct"] = sum(valid) / len(valid) if valid else 0.0

    fieldnames = ["scenario_id", "country", "gold", "story_words", "rot_words", "cluster", "mean_correct", *columns]
    write_csv(output_dir / "scenario_clusters.csv", rows, fieldnames)

    summary = {}
    for cluster_id in sorted(set(labels)):
        cluster_rows = [row for row in rows if row["cluster"] == int(cluster_id)]
        summary[str(int(cluster_id))] = {
            "n": len(cluster_rows),
            "mean_correct": sum(row["mean_correct"] for row in cluster_rows) / len(cluster_rows),
            "top_countries": Counter(row["country"] for row in cluster_rows).most_common(8),
            "gold_counts": Counter(row["gold"] for row in cluster_rows),
        }
    return {"status": "ok", "n_clusters": int(n_clusters), "columns": columns, "summary": summary, "models": models}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-run ML analysis over completed cultural debate runs.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec: name=path=model1,model2. Repeat for each result file.",
    )
    parser.add_argument("--output_dir", default="results/analysis_cross_run_ml")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    specs = args.run or [
        "qwen7_qwen15=results/multi/open_qwen7_qwen15_full.jsonl=qwen7,qwen15",
        "qwen7_gemma4=results/multi/open_qwen7_gemma4_full.jsonl=qwen7,gemma4",
        "qwen7_ollama_llama3_8b=results/multi/open_qwen7_ollama_llama3_8b_full.jsonl=qwen7,ollama_llama3_8b",
        "single_qwen7=results/single/single_model/qwen25_7b_without_rot.jsonl=qwen",
        "single_gemma4=results/single/single_model/gemma4_e4b_without_rot.jsonl=gemma4",
        "single_ollama_llama3_8b=results/single/single_model/ollama_llama3_8b_without_rot.jsonl=ollama_llama3_8b",
    ]

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_long_records(specs)
    if not records:
        raise ValueError("No records found for cross-run ML analysis.")

    write_csv(
        output_dir / "cross_run_long_records.csv",
        records,
        [
            "run",
            "source_file",
            "row_index",
            "scenario_id",
            "country",
            "gold",
            "model",
            "field",
            "pred",
            "correct",
            "valid_pred",
            "story_words",
            "rot_words",
            "has_debate_turns",
        ],
    )
    write_csv(
        output_dir / "cross_run_model_metrics.csv",
        model_metrics(records),
        ["run", "model", "n", "accuracy", "valid_rate", "yes_rate", "no_rate", "neutral_rate", "invalid_rate"],
    )
    write_csv(
        output_dir / "cross_run_pairwise_agreement.csv",
        pairwise_agreement(records),
        ["run", "left_model", "right_model", "n", "agreement", "both_correct_rate", "left_only_correct_rate", "right_only_correct_rate"],
    )

    rf_summary = run_random_forest(records, output_dir, args.seed)
    cluster_summary = scenario_clusters(records, output_dir, args.k, args.seed)

    summary = {
        "n_long_records": len(records),
        "runs": sorted({record["run"] for record in records}),
        "models": sorted({record["model"] for record in records}),
        "random_forest": rf_summary,
        "scenario_clusters": cluster_summary,
    }
    with (output_dir / "cross_run_ml_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved cross-run ML analysis to {output_dir}")


if __name__ == "__main__":
    main()
