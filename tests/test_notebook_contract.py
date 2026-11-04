import json
from pathlib import Path

import unittest


class TestNotebookContract(unittest.TestCase):
    def test_notebook_references_python_package_entrypoints(self) -> None:
        notebook_path = Path("notebooks/vertical_slice.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        sources = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        self.assertIn("from spus_ft import (", sources)
        self.assertIn("fine_tune,", sources)
        self.assertIn("infer_rollout,", sources)
        self.assertIn("evaluate_rollout,", sources)
