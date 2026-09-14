"""The three-tier rule: data/, evaluation/, and config/ must never import
torch (they run on the CPU-only laptop); training/ imports torch lazily inside
functions so its files stay import-safe/readable without a GPU."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _py_files(*dirs: str):
    for d in dirs:
        for p in sorted((ROOT / d).rglob("*.py")):
            yield p


class TestTorchFreeTiers:
    def test_no_torch_imports_in_data_evaluation_config(self):
        bad = []
        for p in _py_files("data", "evaluation", "config"):
            text = p.read_text()
            if re.search(r"^\s*(import torch|from torch)", text, re.MULTILINE):
                bad.append(str(p))
        assert not bad, f"torch imported in: {bad}"


class TestTrainingTierLazyTorch:
    def test_training_files_have_no_module_level_torch_imports(self):
        bad = []
        for p in _py_files("training"):
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if re.match(r"^import torch|^from torch", line):
                    bad.append(f"{p}:{i}")
        assert not bad, f"module-level torch imports in: {bad}"
