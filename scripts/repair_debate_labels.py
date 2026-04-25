import argparse
import json
from pathlib import Path
from typing import Iterable, List

from label_utils import extract_label


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def infer_models(row: dict) -> List[str]:
    models = []
    for key in row:
        if key.endswith("_final_raw"):
            models.append(key[: -len("_final_raw")])
    return sorted(models)


def infer_models_from_rows(rows: Iterable[dict]) -> List[str]:
    models = set()
    for row in rows:
        models.update(infer_models(row))
    return sorted(models)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair normalized debate labels from stored *_final_raw outputs.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--models", nargs="*", default=[], help="Optional list of model aliases to repair.")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    rows = list(read_jsonl(input_file))
    if not rows:
        raise ValueError("Input file contains no rows.")

    models = args.models or infer_models_from_rows(rows)
    if not models:
        raise ValueError("Could not infer any model aliases from *_final_raw fields.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    changes = {model: 0 for model in models}
    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            repaired = dict(row)
            for model in models:
                raw_key = f"{model}_final_raw"
                final_key = f"{model}_final"
                if raw_key not in repaired:
                    continue
                new_label = extract_label(str(repaired.get(raw_key, "")))
                if repaired.get(final_key) != new_label:
                    changes[model] += 1
                repaired[final_key] = new_label
            f.write(json.dumps(repaired, ensure_ascii=False) + "\n")

    print(f"Repaired {len(rows)} rows from {input_file} -> {output_file}")
    for model in models:
        print(f"{model}: updated {changes[model]} normalized labels")


if __name__ == "__main__":
    main()
