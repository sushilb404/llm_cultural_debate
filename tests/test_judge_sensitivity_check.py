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

import judge_sensitivity_check


class JudgeSensitivityTests(unittest.TestCase):
    def test_neither_and_neutral_do_not_disagree_by_themselves(self):
        neither_text = "assistant\nLabel: Neither\nReason: Mixed context."
        neutral_text = "assistant\nLabel: neutral\nReason: Mixed context."

        self.assertEqual(judge_sensitivity_check.judge_base(neither_text), "neutral")
        self.assertEqual(judge_sensitivity_check.judge_alternate(neither_text), "neutral")
        self.assertEqual(judge_sensitivity_check.judge_base(neutral_text), "neutral")
        self.assertEqual(judge_sensitivity_check.judge_alternate(neutral_text), "neutral")

    def test_malformed_output_can_disagree_under_stricter_policy(self):
        text = "assistant\nThe action is socially acceptable.\nReason: It follows the rule."

        self.assertEqual(judge_sensitivity_check.judge_base(text), "yes")
        self.assertEqual(judge_sensitivity_check.judge_alternate(text), "invalid")

    def test_external_alternate_file_defaults_to_normalized_labels(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_file = tmp_path / "input.jsonl"
            alternate_file = tmp_path / "alternate.jsonl"
            output_json = tmp_path / "summary.json"
            output_csv = tmp_path / "disagreements.csv"

            input_file.write_text(
                (
                    '{"Country":"egypt","Story":"story","Rule-of-Thumb":"rule","Gold Label":"no",'
                    '"model_final_raw":"assistant\\nLabel: Yes\\nReason: wrong.","model_final":"yes"}\n'
                ),
                encoding="utf-8",
            )
            alternate_file.write_text(
                (
                    '{"Country":"egypt","Story":"story","Rule-of-Thumb":"rule","Gold Label":"no",'
                    '"model_final_raw":"assistant\\nLabel: Yes\\nReason: stale raw.","model_final":"no"}\n'
                ),
                encoding="utf-8",
            )

            with patch.object(
                judge_sensitivity_check.argparse.ArgumentParser,
                "parse_args",
                return_value=Namespace(
                    input_file=str(input_file),
                    model="model",
                    alternate_file=str(alternate_file),
                    alternate_field="",
                    output_json=str(output_json),
                    output_csv=str(output_csv),
                ),
            ):
                judge_sensitivity_check.main()

            summary = judge_sensitivity_check.json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["agreement_rate"], 0.0)
            self.assertEqual(summary["accuracy_base"], 0.0)
            self.assertEqual(summary["accuracy_alternate"], 1.0)


if __name__ == "__main__":
    unittest.main()
