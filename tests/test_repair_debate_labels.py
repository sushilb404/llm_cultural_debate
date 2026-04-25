import sys
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import repair_debate_labels


class RepairDebateLabelsTests(unittest.TestCase):
    def test_infers_models_from_later_rows_when_first_row_has_error(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_file = tmp_path / "input.jsonl"
            output_file = tmp_path / "output.jsonl"

            input_file.write_text(
                (
                    '{"Country":"egypt","error":"generation failed"}\n'
                    '{"Country":"egypt","error":"","model_a_final_raw":"assistant\\nLabel: Yes\\nReason: ok.",'
                    '"model_a_final":"invalid"}\n'
                ),
                encoding="utf-8",
            )

            with patch.object(
                repair_debate_labels.argparse.ArgumentParser,
                "parse_args",
                return_value=Namespace(
                    input_file=str(input_file),
                    output_file=str(output_file),
                    models=[],
                ),
            ):
                repair_debate_labels.main()

            repaired_rows = [
                repair_debate_labels.json.loads(line)
                for line in output_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(repaired_rows[0]["error"], "generation failed")
            self.assertEqual(repaired_rows[1]["model_a_final"], "yes")


if __name__ == "__main__":
    unittest.main()
