import unittest
from unittest.mock import patch

from spus_ft.runtime import check_spus_runtime_dependencies, require_spus_runtime_dependencies


class TestRuntime(unittest.TestCase):
    def test_check_spus_runtime_dependencies_reports_versions(self) -> None:
        with patch("spus_ft.runtime.import_module") as mock_import_module, patch(
            "spus_ft.runtime.version"
        ) as mock_version:
            mock_import_module.side_effect = [object(), object(), object()]
            mock_version.side_effect = ["2.6.0", "0.35.0", "0.5.2"]

            result = check_spus_runtime_dependencies()

        self.assertTrue(result["ok"])
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["installed"]["torch"], "2.6.0")
        self.assertEqual(result["installed"]["huggingface_hub"], "0.35.0")
        self.assertEqual(result["installed"]["safetensors"], "0.5.2")

    def test_require_spus_runtime_dependencies_raises_for_missing_packages(self) -> None:
        with patch("spus_ft.runtime.check_spus_runtime_dependencies") as mock_check:
            mock_check.return_value = {
                "ok": False,
                "installed": {},
                "missing": ["torch"],
            }

            with self.assertRaises(RuntimeError) as error:
                require_spus_runtime_dependencies()

        self.assertIn("uv sync --extra spus", str(error.exception))
