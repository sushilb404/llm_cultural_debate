import importlib
import unittest


resilient_resume_runner = importlib.import_module("scripts.resilient_resume_runner")


class ResilientResumeRunnerTests(unittest.TestCase):
    def test_configuration_exit_is_not_retriable(self):
        self.assertFalse(
            resilient_resume_runner.should_restart_worker(
                resilient_resume_runner.NON_RETRIABLE_CONFIG_ERROR_EXIT_CODE
            )
        )

    def test_generic_worker_failure_is_retriable(self):
        self.assertTrue(resilient_resume_runner.should_restart_worker(1))


if __name__ == "__main__":
    unittest.main()
