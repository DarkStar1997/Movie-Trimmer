"""SmoothingConfig: override validation and defaults."""

import pytest

from scanner.temporal import SmoothingConfig


def test_defaults_come_from_config():
    from scanner import config
    cfg = SmoothingConfig()
    for k, v in config.DEFAULT_SMOOTHING.items():
        assert getattr(cfg, k) == v


def test_overrides_apply():
    cfg = SmoothingConfig.from_overrides({"pad_s": 3.5,
                                          "bridge_gap_s": 0.75})
    assert cfg.pad_s == 3.5 and cfg.bridge_gap_s == 0.75


def test_unknown_keys_rejected():
    with pytest.raises(ValueError, match="unknown smoothing keys"):
        SmoothingConfig.from_overrides({"bogus": 1})


def test_none_overrides_is_defaults():
    assert SmoothingConfig.from_overrides(None) == SmoothingConfig()
