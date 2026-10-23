from pathlib import Path
import unittest

from spus_ft.paths import TutorialPaths
from spus_ft.pipeline import run_synthetic_demo


class TestPipeline(unittest.TestCase):
    def test_pipeline_writes_outputs_to_run_directory(self) -> None:
        tmp_path = Path("/tmp/src-test-pipeline")
        if tmp_path.exists():
            for child in sorted(tmp_path.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            tmp_path.rmdir()
        tmp_path.mkdir(parents=True)
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        (repo_root / "configs").mkdir()
        (repo_root / "configs" / "synthetic_demo.json").write_text(
            """
            {
              "num_samples": 2,
              "num_timesteps": 3,
              "num_features": 2,
              "decay": 0.9,
              "forcing": 0.1,
              "baseline_alpha": 0.8,
              "seed": 11
            }
            """.strip(),
            encoding="utf-8",
        )
        paths = TutorialPaths.discover(repo_root)

        outcome = run_synthetic_demo(paths=paths)

        self.assertEqual(outcome["run_dir"], paths.runs_root / "phase1_synthetic_demo")
        self.assertTrue((outcome["run_dir"] / "rollout.json").is_file())
        self.assertTrue((outcome["run_dir"] / "summary.json").is_file())
