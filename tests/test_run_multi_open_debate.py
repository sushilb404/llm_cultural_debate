import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

torch_stub = types.ModuleType("torch")
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules.setdefault("torch", torch_stub)

transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoModelForCausalLM = object
transformers_stub.AutoTokenizer = object
transformers_stub.BitsAndBytesConfig = object
sys.modules.setdefault("transformers", transformers_stub)

run_multi_open_debate = importlib.import_module("scripts.run_multi_open_debate")


class ValidateGpuCapacityTests(unittest.TestCase):
    def test_raises_clear_error_for_two_7b_models_on_15gb_gpu(self):
        with self.assertRaisesRegex(
            run_multi_open_debate.PreflightConfigurationError,
            "does not have enough VRAM.*--load_in_4bit.*smaller models",
        ):
            run_multi_open_debate.validate_gpu_capacity(
                model_ids=[
                    "Qwen/Qwen2.5-7B-Instruct",
                    "Qwen/Qwen2.5-7B-Instruct",
                ],
                load_in_4bit=False,
                gpu_memory_gb_by_device=[14.56],
            )

    def test_visible_gpu_memory_is_listed_per_device(self):
        run_multi_open_debate.torch.cuda.device_count = lambda: 2
        run_multi_open_debate.torch.cuda.get_device_properties = lambda index: types.SimpleNamespace(
            total_memory=16 * (1024 ** 3)
        )

        memory_gb_by_device = run_multi_open_debate.get_visible_gpu_memory_gb_by_device()

        self.assertEqual(memory_gb_by_device, [16.0, 16.0])

    def test_allows_two_7b_models_when_each_fits_on_its_own_gpu(self):
        run_multi_open_debate.validate_gpu_capacity(
            model_ids=[
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
            ],
            load_in_4bit=False,
            gpu_memory_gb_by_device=[16.0, 16.0],
        )


class DebateOneParsingTests(unittest.TestCase):
    def test_final_labels_are_parsed_from_raw_generation_without_keyword_stripping(self):
        final_a_raw = (
            "Return exactly two lines using this format:\n"
            "Label: <one of Yes, No, Neither>\n"
            "Reason: one short sentence.\n"
            "Label:\n"
            "assistant\n"
            "No\n"
            "Reason: The rule does not support the action."
        )
        final_b_raw = (
            "Return exactly two lines using this format:\n"
            "Label: <one of Yes, No, Neither>\n"
            "Reason: one short sentence.\n"
            "Label:\n"
            "assistant\n"
            "Label: Yes\n"
            "Reason: The story follows the rule."
        )

        with patch.object(
            run_multi_open_debate,
            "generate",
            side_effect=[
                "Answer: Yes\nBecause it follows the rule.",
                "Answer: No\nBecause it conflicts with the rule.",
                "Response: I disagree.",
                "Response: I disagree as well.",
                final_a_raw,
                final_b_raw,
            ],
        ):
            out = run_multi_open_debate.debate_one(
                record={
                    "Country": "united_states_of_america",
                    "Story": "A short story.",
                    "Rule-of-Thumb": "A short rule.",
                },
                model_a=(object(), object()),
                model_b=(object(), object()),
                alias_a="model_a",
                alias_b="model_b",
                max_new_tokens=32,
            )

        self.assertEqual(out["model_a_final_raw"], final_a_raw)
        self.assertEqual(out["model_b_final_raw"], final_b_raw)
        self.assertEqual(out["model_a_final"], "no")
        self.assertEqual(out["model_b_final"], "yes")


if __name__ == "__main__":
    unittest.main()
