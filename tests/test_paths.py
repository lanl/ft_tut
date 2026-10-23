from pathlib import Path
import unittest

from spus_ft.paths import TutorialPaths


class TestPaths(unittest.TestCase):
    def test_paths_default_to_external_sibling_directories(self) -> None:
        tmp_path = Path("/tmp/src-test-paths-default")
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

        paths = TutorialPaths.discover(repo_root)

        self.assertEqual(paths.repo_root, repo_root.resolve())
        self.assertEqual(paths.aux_root, (tmp_path / "tut-aux").resolve())
        self.assertEqual(paths.data_root, (tmp_path / "tut-aux" / "spus-ft-data").resolve())
        self.assertEqual(paths.models_root, (tmp_path / "tut-aux" / "spus-models").resolve())
        self.assertEqual(paths.runs_root, (tmp_path / "tut-aux" / "spus-ft-runs").resolve())
        self.assertEqual(paths.cache_root, (tmp_path / "tut-aux" / "spus-ft-cache").resolve())


    def test_ensure_runtime_dirs_creates_external_roots(self) -> None:
        tmp_path = Path("/tmp/src-test-paths-runtime")
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
        paths = TutorialPaths.discover(repo_root)

        paths.ensure_runtime_dirs()

        self.assertTrue(paths.aux_root.is_dir())
        self.assertTrue(paths.data_root.is_dir())
        self.assertTrue(paths.models_root.is_dir())
        self.assertTrue(paths.runs_root.is_dir())
        self.assertTrue(paths.cache_root.is_dir())
