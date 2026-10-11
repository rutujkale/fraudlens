"""Tests for the config layer (Part 01, task 1G)."""

import pytest
import yaml

from fraudlens import config as cfg_module
from fraudlens.config import config_hash, load_config

PROFILES = ["tiny", "dev", "main", "drift_lab", "unseen_lab"]
EMBARGO_BY_PROFILE = {"tiny": 3, "dev": 7, "main": 21, "drift_lab": 14, "unseen_lab": 14}


def test_every_profile_loads():
    for profile in PROFILES:
        cfg = load_config(profile)
        assert cfg.simulation.profile == profile
        assert cfg.simulation.customers > 0
        assert cfg.simulation.days > 0


@pytest.mark.parametrize("profile", PROFILES)
def test_scenario_mixes_sum_to_one(profile):
    cfg = load_config(profile)
    for name in ("scenario_mix_period_a", "scenario_mix_period_b"):
        mix = getattr(cfg.simulation, name)
        assert abs(sum(mix.values()) - 1.0) < 1e-6
    if cfg.simulation.scenario_mix_override:
        override = cfg.simulation.scenario_mix_override
        assert abs(sum(override.values()) - 1.0) < 1e-6


@pytest.mark.parametrize("profile,expected", EMBARGO_BY_PROFILE.items())
def test_embargo_days_is_max_label_delay(profile, expected):
    assert load_config(profile).embargo_days == expected


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        load_config("does_not_exist")


def test_invalid_mix_raises(tmp_path, monkeypatch):
    bad = tmp_path / "config_bad.yaml"
    with open(cfg_module._config_path(), encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    bad_mix = {k: 0.0 for k in raw["common"]["scenario_mix_period_a"]}
    raw["common"]["scenario_mix_period_a"] = bad_mix
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.setenv("FRAUDLENS_CONFIG_PATH", str(bad))
    with pytest.raises(ValueError):
        load_config("dev")


def test_config_hash_stable_across_loads():
    assert config_hash(load_config("dev")) == config_hash(load_config("dev"))


def test_config_hash_differs_across_profiles():
    assert config_hash(load_config("dev")) != config_hash(load_config("main"))
    assert len(config_hash(load_config("dev"))) == 64