from pathlib import Path
import tempfile
import unittest

from spus_ft.data import (
    FNS_KF_PATTERNS,
    FNS_KF_REPO_ID,
    SPUS_MODEL_PATTERNS,
    SPUS_MODEL_REPO_IDS,
    clone_spus_model_collection,
    clone_spus_model_repo,
    download_fns_kf_dataset,
    download_spus_model,
    download_spus_model_collection,
    resolve_fns_kf_data_dir,
    resolve_spus_model_git_url,
    resolve_spus_model_dir,
    resolve_spus_models_dir,
)
from spus_ft.paths import TutorialPaths


class TestFnsKfData(unittest.TestCase):
    def test_resolve_fns_kf_data_dir_uses_data_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-fns-kf-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)

            resolved = resolve_fns_kf_data_dir(paths)

            self.assertEqual(resolved, paths.data_root / "FNS-KF")

    def test_download_fns_kf_dataset_calls_snapshot_download(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-fns-kf-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)
            calls = {}

            def fake_snapshot_download(**kwargs):
                calls.update(kwargs)
                return kwargs["local_dir"]

            outcome = download_fns_kf_dataset(
                paths=paths,
                snapshot_download_fn=fake_snapshot_download,
            )

            self.assertEqual(outcome["repo_id"], FNS_KF_REPO_ID)
            self.assertEqual(outcome["target_dir"], paths.data_root / "FNS-KF")
            self.assertEqual(calls["repo_id"], FNS_KF_REPO_ID)
            self.assertEqual(calls["repo_type"], "dataset")
            self.assertEqual(calls["allow_patterns"], FNS_KF_PATTERNS)
            self.assertEqual(Path(calls["local_dir"]), paths.data_root / "FNS-KF")

    def test_resolve_spus_models_dir_uses_models_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-models-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)

            resolved = resolve_spus_models_dir(paths)

            self.assertEqual(resolved, paths.models_root)

    def test_download_spus_model_calls_snapshot_download(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-model-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)
            calls = {}
            repo_id = SPUS_MODEL_REPO_IDS[0]

            def fake_snapshot_download(**kwargs):
                calls.update(kwargs)
                return kwargs["local_dir"]

            outcome = download_spus_model(
                repo_id=repo_id,
                paths=paths,
                snapshot_download_fn=fake_snapshot_download,
            )

            self.assertEqual(outcome["repo_id"], repo_id)
            self.assertEqual(outcome["target_dir"], resolve_spus_model_dir(repo_id, paths))
            self.assertEqual(calls["repo_id"], repo_id)
            self.assertEqual(calls["repo_type"], "model")
            self.assertEqual(calls["allow_patterns"], SPUS_MODEL_PATTERNS)
            self.assertEqual(Path(calls["local_dir"]), resolve_spus_model_dir(repo_id, paths))

    def test_download_spus_model_collection_downloads_all_repos(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-model-collection-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)
            seen_repo_ids = []

            def fake_snapshot_download(**kwargs):
                seen_repo_ids.append(kwargs["repo_id"])
                return kwargs["local_dir"]

            outcome = download_spus_model_collection(
                paths=paths,
                snapshot_download_fn=fake_snapshot_download,
            )

            self.assertEqual(outcome["repo_ids"], SPUS_MODEL_REPO_IDS)
            self.assertEqual(seen_repo_ids, SPUS_MODEL_REPO_IDS)
            self.assertEqual(len(outcome["downloads"]), len(SPUS_MODEL_REPO_IDS))

    def test_clone_spus_model_repo_calls_git_clone(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-clone-model-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)
            calls = []
            repo_id = SPUS_MODEL_REPO_IDS[0]

            def fake_git_runner(command, check, capture_output, text):
                calls.append(command)
                Path(command[-1]).mkdir(parents=True, exist_ok=True)
                (Path(command[-1]) / ".git").mkdir(exist_ok=True)

            outcome = clone_spus_model_repo(
                repo_id=repo_id,
                paths=paths,
                local_dir=repo_root / 'models' / repo_id.split('/', 1)[1],
                git_runner=fake_git_runner,
            )

            self.assertEqual(outcome["repo_id"], repo_id)
            self.assertEqual(outcome["repo_url"], resolve_spus_model_git_url(repo_id))
            self.assertTrue(outcome["cloned"])
            self.assertEqual(
                calls[0],
                ["git", "clone", resolve_spus_model_git_url(repo_id), str((repo_root / 'models' / repo_id.split('/', 1)[1]).resolve())],
            )

    def test_clone_spus_model_repo_can_checkout_revision(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-clone-rev-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)
            calls = []
            repo_id = SPUS_MODEL_REPO_IDS[0]
            target_dir = resolve_spus_model_dir(repo_id, paths)
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / ".git").mkdir(exist_ok=True)

            def fake_git_runner(command, check, capture_output, text):
                calls.append(command)

            outcome = clone_spus_model_repo(
                repo_id=repo_id,
                paths=paths,
                local_dir=target_dir,
                revision="main",
                git_runner=fake_git_runner,
            )

            self.assertFalse(outcome["cloned"])
            self.assertEqual(calls[0], ["git", "-C", str(target_dir), "checkout", "main"])

    def test_clone_spus_model_collection_clones_all_repos(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-clone-collection-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)
            seen_commands = []

            def fake_git_runner(command, check, capture_output, text):
                seen_commands.append(command)
                if len(command) >= 4 and command[1] == "clone":
                    Path(command[-1]).mkdir(parents=True, exist_ok=True)
                    (Path(command[-1]) / ".git").mkdir(exist_ok=True)

            outcome = clone_spus_model_collection(
                paths=paths,
                git_runner=fake_git_runner,
                repo_ids=SPUS_MODEL_REPO_IDS,
                local_root=repo_root / 'models',
            )

            self.assertEqual(outcome["repo_ids"], SPUS_MODEL_REPO_IDS)
            self.assertEqual(len(outcome["clones"]), len(SPUS_MODEL_REPO_IDS))
            self.assertEqual(len(seen_commands), len(SPUS_MODEL_REPO_IDS))

