import json
from pathlib import Path

import unittest


class TestNotebookContract(unittest.TestCase):
    def test_notebook_references_python_package_entrypoints(self) -> None:
        notebook_path = Path("notebooks/01_phase0_phase1_scaffold.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        sources = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        self.assertIn("from spus_ft.pipeline import run_synthetic_demo", sources)
        self.assertIn("from spus_ft.paths import TutorialPaths", sources)
