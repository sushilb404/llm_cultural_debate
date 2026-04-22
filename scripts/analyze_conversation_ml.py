import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from label_utils import normalize_label


LABEL_TO_ID = {"invalid": -1, "no": 0, "neutral": 1, "yes": 2}


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_models(sample: dict) -> List[str]:
    models = []
    for key in sample:
        if key.endswith("_final"):
            models.append(key[:-6])
    return sorted(set(models))


def word_count(text: str) -> int:
    return len((text or "").split())


def build_features(records: List[dict], models: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for model in models:
        feats: List[List[float]] = []
        targets: List[int] = []
        for row in records:
            initial = normalize_label(str(row.get(f"{model}_1", "")))
            final = normalize_label(str(row.get(f"{model}_final", "")))
            gold = normalize_label(str(row.get("Gold Label", "")))
            feedback = str(row.get(f"{model}_2", ""))

            x = [
                LABEL_TO_ID.get(initial, -1),
                int(initial == gold),
                int(final == gold),
                int(initial != final),
                word_count(str(row.get(f"{model}_1", ""))),
                word_count(feedback),
                word_count(str(row.get("Story", ""))),
                word_count(str(row.get("Rule-of-Thumb", ""))),
            ]
            # Target: whether stance changed in this conversation.
            y = int(initial != final)
            feats.append(x)
            targets.append(y)

        out[f"{model}_X"] = np.asarray(feats, dtype=float)
        out[f"{model}_y"] = np.asarray(targets, dtype=int)
    return out


def run_kmeans(X: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    if len(X) < n_clusters:
        n_clusters = max(1, len(X))
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = model.fit_predict(X)
    return labels


def run_random_forest(X: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, object]:
    if len(np.unique(y)) < 2:
        return {
            "status": "skipped",
            "reason": "target has fewer than 2 classes",
            "feature_importance": [],
            "report": {},
        }

    rf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced")
    rf.fit(X, y)
    pred = rf.predict(X)

    report = classification_report(y, pred, output_dict=True, zero_division=0)
    return {
        "status": "ok",
        "feature_importance": rf.feature_importances_.tolist(),
        "report": report,
    }


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K-means and Random Forest on debate conversation features.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", default="results/analysis_ml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    records = read_jsonl(input_file)
    if not records:
        raise ValueError("Input file has no records.")

    models = args.models if args.models else infer_models(records[0])
    if not models:
        raise ValueError("Could not infer model aliases from *_final fields.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = build_features(records, models)
    feature_names = [
        "initial_label_id",
        "initial_correct",
        "final_correct",
        "stance_changed",
        "initial_len",
        "feedback_len",
        "story_len",
        "rot_len",
    ]

    summary = {
        "input_file": str(input_file),
        "models": {},
    }

    for model in models:
        X = features[f"{model}_X"]
        y = features[f"{model}_y"]

        cluster_labels = run_kmeans(X, n_clusters=args.k, seed=args.seed)
        rf_result = run_random_forest(X, y, seed=args.seed)

        cluster_rows = []
        for idx, row in enumerate(records):
            cluster_rows.append(
                {
                    "row_index": idx,
                    "country": row.get("Country", ""),
                    "gold": normalize_label(str(row.get("Gold Label", ""))),
                    "initial": normalize_label(str(row.get(f"{model}_1", ""))),
                    "final": normalize_label(str(row.get(f"{model}_final", ""))),
                    "cluster": int(cluster_labels[idx]),
                }
            )

        write_csv(
            output_dir / f"{model}_kmeans_clusters.csv",
            cluster_rows,
            ["row_index", "country", "gold", "initial", "final", "cluster"],
        )

        importance_rows = []
        if rf_result.get("status") == "ok":
            for name, val in zip(feature_names, rf_result["feature_importance"]):
                importance_rows.append({"feature": name, "importance": float(val)})
            write_csv(output_dir / f"{model}_rf_feature_importance.csv", importance_rows, ["feature", "importance"])

        summary["models"][model] = {
            "n_rows": int(len(X)),
            "n_clusters": int(min(args.k, len(X))),
            "rf": rf_result,
            "cluster_counts": {
                str(int(c)): int((cluster_labels == c).sum()) for c in np.unique(cluster_labels)
            },
        }

    summary_path = output_dir / "ml_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved ML summary: {summary_path}")


if __name__ == "__main__":
    main()
