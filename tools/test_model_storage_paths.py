"""Regression tests for automatic model/download directory resolution."""

import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SERVER = Path(__file__).resolve().parents[1] / "strata_moeboost_server.py"
TREE = ast.parse(SERVER.read_text(encoding="utf-8"))
FUNCTION_NAMES = {
    "_expand_user_path",
    "_has_model_files",
    "get_model_storage_dir",
    "get_model_search_dirs",
    "find_default_model_file",
}
FUNCTIONS = [
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_NAMES
]


def build_helpers(root: Path):
    namespace = {"os": os, "Path": Path, "ROOT": root}
    exec(compile(ast.Module(body=FUNCTIONS, type_ignores=[]), str(SERVER), "exec"), namespace)
    return namespace


class ModelStoragePathTests(unittest.TestCase):
    def test_default_is_home_strata_moeboost_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            helpers = build_helpers(home / "project")
            with patch.object(Path, "home", return_value=home), patch.dict(os.environ, {}, clear=False):
                os.environ.pop("STRATA_MODEL_DIR", None)
                self.assertEqual(helpers["get_model_storage_dir"](), home / "strata-moeboost" / "model")

    def test_existing_legacy_model_directory_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            legacy = home / "project" / "models"
            legacy.mkdir(parents=True)
            (legacy / "existing-moe.gguf").write_bytes(b"model")
            helpers = build_helpers(home / "project")
            with patch.object(Path, "home", return_value=home), patch.dict(os.environ, {}, clear=False):
                os.environ.pop("STRATA_MODEL_DIR", None)
                self.assertEqual(helpers["get_model_storage_dir"](), legacy.resolve())
                self.assertEqual(helpers["find_default_model_file"](), legacy / "existing-moe.gguf")

    def test_environment_override_expands_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            helpers = build_helpers(home / "project")
            with patch.object(Path, "home", return_value=home), patch.dict(os.environ, {"STRATA_MODEL_DIR": "~/custom-models", "HOME": str(home)}):
                self.assertEqual(helpers["get_model_storage_dir"](), home / "custom-models")


if __name__ == "__main__":
    unittest.main()
