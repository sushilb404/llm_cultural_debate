import importlib
import sys
import types
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
