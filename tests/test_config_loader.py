"""Tests for memedit.utils.config_loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from memedit.core.config import MemEditConfig
from memedit.utils.config_loader import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"
TINY_YAML = REPO_ROOT / "configs" / "tiny.yaml"


class TestLoadConfig:
    @pytest.mark.skipif(not DEFAULT_YAML.exists(), reason="default.yaml missing")
    def test_loads_default(self):
        cfg = load_config(DEFAULT_YAML)
        assert isinstance(cfg, MemEditConfig)
        # A couple of sanity checks against the YAML values.
        assert cfg.memory.num_layers == 4
        assert cfg.memory.hidden_dim == 4096
        assert cfg.attribution.riemann_steps == 20
        assert cfg.attribution.sparsity_tau == pytest.approx(0.03)
        assert cfg.delete.num_steps == 5
        assert cfg.mome.initial_num_shards == 2

    @pytest.mark.skipif(not TINY_YAML.exists(), reason="tiny.yaml missing")
    def test_loads_tiny(self):
        cfg = load_config(TINY_YAML)
        assert cfg.memory.hidden_dim == 64
        assert cfg.memory.vocab_size == 256
        assert cfg.mome.expansion_threshold == pytest.approx(0.3)

    def test_loads_arbitrary_yaml(self, tmp_path):
        yaml_path = tmp_path / "c.yaml"
        yaml_path.write_text(
            "memory:\n"
            "  hidden_dim: 128\n"
            "  vocab_size: 500\n"
            "device: cpu\n"
            "seed: 7\n"
        )
        cfg = load_config(yaml_path)
        assert cfg.memory.hidden_dim == 128
        assert cfg.memory.vocab_size == 500
        assert cfg.seed == 7
