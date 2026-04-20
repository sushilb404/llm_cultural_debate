import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


REQUIRED_KEYS = {"Country", "Story", "Rule-of-Thumb", "Gold Label"}


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def scenario_key(row: dict) -> Tuple[str, str, str, str]:
    return (
        str(row.get("Country", "")),
        str(row.get("Story", "")),
        str(row.get("Rule-of-Thumb", "")),
        str(row.get("Gold Label", "")),
    )


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate debate output integrity (duplicates, required keys, reference coverage).")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--reference_file", default="data/normad.jsonl")
    parser.add_argument("--expected_rows", type=int, default=0)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--duplicates_csv", default="")
    parser.add_argument("--rewrite_deduped", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero status if any integrity violation is found.")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    reference_file = Path(args.reference_file)
    if not reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_file}")

    output_json = (
        Path(args.output_json)
        if args.output_json
        else input_file.with_name(input_file.stem + "_integrity_report.json")
    )
    duplicates_csv = (
        Path(args.duplicates_csv)
        if args.duplicates_csv
        else input_file.with_name(input_file.stem + "_duplicate_rows.csv")
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    duplicates_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_file)
    reference_rows = read_jsonl(reference_file)
    reference_keys: Set[Tuple[str, str, str, str]] = {scenario_key(r) for r in reference_rows}

    missing_required: List[dict] = []
    duplicate_rows: List[dict] = []
    seen: Dict[Tuple[str, str, str, str], int] = {}
    deduped: List[dict] = []

    for idx, row in enumerate(rows):
        missing = [k for k in REQUIRED_KEYS if not str(row.get(k, "")).strip()]
        if missing:
            missing_required.append({"row_index": idx, "missing_keys": ";".join(sorted(missing))})

        key = scenario_key(row)
        if key in seen:
            duplicate_rows.append(
                {
                    "row_index": idx,
                    "duplicate_of": seen[key],
                    "country": key[0],
                    "story_preview": key[1][:120],
                }
            )
            continue

        seen[key] = idx
        deduped.append(row)

    deduped_keys = set(seen.keys())
    unexpected_keys = deduped_keys - reference_keys
    missing_from_output = reference_keys - deduped_keys

    if duplicate_rows:
        write_csv(duplicates_csv, duplicate_rows, ["row_index", "duplicate_of", "country", "story_preview"])

    if args.rewrite_deduped and duplicate_rows:
        backup = input_file.with_suffix(input_file.suffix + ".pre_integrity_dedupe.bak")
        if not backup.exists():
            input_file.replace(backup)
        else:
            input_file.unlink(missing_ok=True)

        with input_file.open("w", encoding="utf-8") as f:
            for row in deduped:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    violations = {
        "duplicates": len(duplicate_rows),
        "missing_required_rows": len(missing_required),
        "unexpected_rows_vs_reference": len(unexpected_keys),
        "missing_reference_rows": len(missing_from_output),
    }

    payload = {
        "input_file": str(input_file),
        "reference_file": str(reference_file),
        "rows_total": len(rows),
        "rows_unique": len(deduped),
        "expected_rows": args.expected_rows,
        "expected_rows_met": (len(deduped) >= args.expected_rows) if args.expected_rows > 0 else None,
        "violations": violations,
        "duplicates_csv": str(duplicates_csv) if duplicate_rows else "",
        "missing_required_examples": missing_required[:20],
        "unexpected_rows_examples": [
            {"country": k[0], "story_preview": k[1][:120]} for k in list(unexpected_keys)[:20]
        ],
        "missing_reference_examples": [
            {"country": k[0], "story_preview": k[1][:120]} for k in list(missing_from_output)[:20]
        ],
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved integrity report: {output_json}")
    if duplicate_rows:
        print(f"Saved duplicate rows: {duplicates_csv}")
    print(f"Unique rows: {len(deduped)} / Total rows: {len(rows)}")

    if args.strict and any(v > 0 for v in violations.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
