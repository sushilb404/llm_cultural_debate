import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_label(text: str) -> str:
    cleaned = (text or "").replace(",", "").replace(".", "").strip().lower()
    if "yes" in cleaned or "is socially acceptable" in cleaned or "are socially acceptable" in cleaned:
        return "yes"
    if "no" in cleaned or "is not socially acceptable" in cleaned or "are not socially acceptable" in cleaned:
        return "no"
    if "neither" in cleaned:
        return "neutral"
    return cleaned


def evaluate_single(input_file: Path, model: str) -> Tuple[List[dict], float]:
    per_country_correct: Dict[str, int] = defaultdict(int)
    per_country_total: Dict[str, int] = defaultdict(int)
    total = 0
    correct = 0

    for row in read_jsonl(input_file):
        country = row.get("Country", "unknown")
        pred = normalize_label(str(row.get(model, "")))
        gold = str(row.get("Gold Label", "")).strip().lower()

        total += 1
        per_country_total[country] += 1
        if pred == gold:
            correct += 1
            per_country_correct[country] += 1

    accuracy = (correct / total) if total else 0.0
    results = [
        {
            "Country": "total",
            "Total": total,
            "Correct": correct,
            "Accuracy": round(accuracy, 4),
        }
    ]

    for country in sorted(per_country_total):
        c_total = per_country_total[country]
        c_correct = per_country_correct[country]
        c_acc = (c_correct / c_total) if c_total else 0.0
        results.append(
            {
                "Country": country,
                "Total": c_total,
                "Correct": c_correct,
                "Accuracy": round(c_acc, 4),
            }
        )

    return results, accuracy


def evaluate_multi(input_file: Path, first_model: str, second_model: str) -> Tuple[List[dict], float, float]:
    first_field = f"{first_model}_final"
    second_field = f"{second_model}_final"

    per_country_correct_1: Dict[str, int] = defaultdict(int)
    per_country_total_1: Dict[str, int] = defaultdict(int)
    per_country_correct_2: Dict[str, int] = defaultdict(int)
    per_country_total_2: Dict[str, int] = defaultdict(int)

    total = 0
    correct_1 = 0
    correct_2 = 0

    for row in read_jsonl(input_file):
        country = row.get("Country", "unknown")
        gold = str(row.get("Gold Label", "")).strip().lower()

        pred_1 = normalize_label(str(row.get(first_field, "")))
        pred_2 = normalize_label(str(row.get(second_field, "")))

        total += 1
        per_country_total_1[country] += 1
        per_country_total_2[country] += 1

        if pred_1 == gold:
            correct_1 += 1
            per_country_correct_1[country] += 1

        if pred_2 == gold:
            correct_2 += 1
            per_country_correct_2[country] += 1

    acc_1 = (correct_1 / total) if total else 0.0
    acc_2 = (correct_2 / total) if total else 0.0

    results = [
        {
            "Country": "total",
            "Total": total,
            f"{first_model}_correct": correct_1,
            f"{first_model}_accuracy": round(acc_1, 4),
            f"{second_model}_correct": correct_2,
            f"{second_model}_accuracy": round(acc_2, 4),
        }
    ]

    all_countries = sorted(set(per_country_total_1) | set(per_country_total_2))
    for country in all_countries:
        total_1 = per_country_total_1[country]
        total_2 = per_country_total_2[country]
        correct_country_1 = per_country_correct_1[country]
        correct_country_2 = per_country_correct_2[country]
        results.append(
            {
                "Country": country,
                f"{first_model}_total": total_1,
                f"{first_model}_correct": correct_country_1,
                f"{first_model}_accuracy": round((correct_country_1 / total_1) if total_1 else 0.0, 4),
                f"{second_model}_total": total_2,
                f"{second_model}_correct": correct_country_2,
                f"{second_model}_accuracy": round((correct_country_2 / total_2) if total_2 else 0.0, 4),
            }
        )

    return results, acc_1, acc_2


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate output JSONL files from this repository.")
    parser.add_argument("--mode", choices=["single", "multi"], required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", default="")
    parser.add_argument("--model", default="", help="For --mode single")
    parser.add_argument("--first_model", default="", help="For --mode multi")
    parser.add_argument("--second_model", default="", help="For --mode multi")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = input_file.with_name(input_file.stem + "_accuracy.jsonl")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        if not args.model:
            raise ValueError("--model is required for --mode single")
        rows, acc = evaluate_single(input_file, args.model)
        write_jsonl(output_file, rows)
        print(f"{args.model} accuracy: {acc:.4f}")
    else:
        if not args.first_model or not args.second_model:
            raise ValueError("--first_model and --second_model are required for --mode multi")
        rows, acc_1, acc_2 = evaluate_multi(input_file, args.first_model, args.second_model)
        write_jsonl(output_file, rows)
        print(f"{args.first_model} accuracy: {acc_1:.4f}")
        print(f"{args.second_model} accuracy: {acc_2:.4f}")

    print(f"Saved evaluation to: {output_file}")


if __name__ == "__main__":
    main()
