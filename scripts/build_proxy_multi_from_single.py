import argparse
import json
from pathlib import Path
from typing import List


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_prediction(row: dict, model_name: str) -> str:
    if f"{model_name}_final" in row:
        return str(row[f"{model_name}_final"])
    if model_name in row:
        return str(row[model_name])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a proxy multi-model debate JSONL from two single-model JSONL files.")
    parser.add_argument("--left_file", required=True)
    parser.add_argument("--right_file", required=True)
    parser.add_argument("--left_model", required=True)
    parser.add_argument("--right_model", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    left_rows = read_jsonl(Path(args.left_file))
    right_rows = read_jsonl(Path(args.right_file))

    n = min(len(left_rows), len(right_rows))
    if n == 0:
        raise ValueError("No rows to merge.")

    out_rows: List[dict] = []
    for i in range(n):
        l = left_rows[i]
        r = right_rows[i]
        out_rows.append(
            {
                "Country": l.get("Country", r.get("Country", "")),
                "Story": l.get("Story", r.get("Story", "")),
                "Rule-of-Thumb": l.get("Rule-of-Thumb", r.get("Rule-of-Thumb", "")),
                "Gold Label": l.get("Gold Label", r.get("Gold Label", "")),
                f"{args.left_model}_final": get_prediction(l, args.left_model),
                f"{args.right_model}_final": get_prediction(r, args.right_model),
            }
        )

    write_jsonl(Path(args.output_file), out_rows)
    print(f"Merged {n} rows into: {args.output_file}")


if __name__ == "__main__":
    main()
