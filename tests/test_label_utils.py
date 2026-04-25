import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from label_utils import extract_label, normalize_label


class LabelUtilsTests(unittest.TestCase):
    def test_assistant_prefixed_bare_label_parses_as_no(self):
        text = "assistant\nNo\nReason: The action conflicts with the rule."
        self.assertEqual(extract_label(text), "no")

    def test_assistant_prefixed_inline_bare_label_with_reason_parses_as_no(self):
        text = "assistant\nNo: The action conflicts with the rule.\nReason: It breaks the norm."
        self.assertEqual(extract_label(text), "no")

    def test_assistant_prefixed_structured_label_parses_as_yes(self):
        text = "assistant\nLabel: Yes\nReason: The action follows the rule."
        self.assertEqual(extract_label(text), "yes")

    def test_assistant_prefixed_multiline_label_marker_parses_as_yes(self):
        text = "assistant\nLabel:\nYes\nReason: The action follows the rule."
        self.assertEqual(extract_label(text), "yes")

    def test_assistant_prefixed_yes_sentence_parses_as_yes(self):
        text = "assistant\nYes, the action follows the rule and is socially acceptable."
        self.assertEqual(extract_label(text), "yes")

    def test_sentence_with_preamble_still_parses_explicit_decision(self):
        text = "assistant\nTherefore, the final answer is Yes."
        self.assertEqual(extract_label(text), "yes")

    def test_prompt_echo_with_trailing_label_still_recovers_real_decision(self):
        text = (
            "Return exactly two lines using this format:\n"
            "Label: <one of Yes, No, Neither>\n"
            "Reason: one short sentence.\n"
            "Label:\n"
            "assistant\n"
            "Label: Yes\n"
            "Reason: It follows the rule."
        )
        self.assertEqual(extract_label(text), "yes")

    def test_neither_and_neutral_both_normalize_to_neutral(self):
        self.assertEqual(normalize_label("Neither"), "neutral")
        self.assertEqual(normalize_label("neutral"), "neutral")

    def test_option_list_echo_remains_invalid(self):
        text = "Label: <one of Yes, No, Neither>"
        self.assertEqual(extract_label(text), "invalid")


if __name__ == "__main__":
    unittest.main()
