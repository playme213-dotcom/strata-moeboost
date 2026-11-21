"""Regression tests for external-provider model ID selection."""

import ast
import unittest
from pathlib import Path


SERVER = Path(__file__).resolve().parents[1] / "strata_moeboost_server.py"
TREE = ast.parse(SERVER.read_text(encoding="utf-8"))
SELECTOR_NODE = next(
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "_select_provider_model"
)
NAMESPACE = {}
exec(compile(ast.Module(body=[SELECTOR_NODE], type_ignores=[]), str(SERVER), "exec"), NAMESPACE)
select_provider_model = NAMESPACE["_select_provider_model"]


class ProviderModelRoutingTests(unittest.TestCase):
    def test_configured_provider_model_beats_gui_display_label(self):
        self.assertEqual(
            select_provider_model("[API] Qwen_Qwen3-Coder-Next-IQ2_XXS", "Qwen_Qwen3-Coder-Next-IQ2_XXS"),
            "Qwen_Qwen3-Coder-Next-IQ2_XXS",
        )

    def test_requested_model_is_fallback_when_provider_model_is_unset(self):
        self.assertEqual(select_provider_model("local-model-id", ""), "local-model-id")

    def test_default_is_used_when_both_model_ids_are_missing(self):
        self.assertEqual(select_provider_model(None, None), "default")


if __name__ == "__main__":
    unittest.main()
