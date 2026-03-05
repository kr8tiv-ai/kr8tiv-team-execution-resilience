from pathlib import Path

import yaml


def test_bot_runtime_model_policy_matches_canonical_lanes() -> None:
    data = yaml.safe_load(Path("config/model-policy.public.yaml").read_text(encoding="utf-8"))
    agents = data["agents"]
    expected_models = {
        "friday": "claude-cli/claude-opus-4-6",
        "arsenal": "codex-cli/gpt-5.3-codex",
        "jocasta": "nvidia/moonshotai/kimi-k2.5",
        "edith": "google-gemini-cli/gemini-3-pro-preview",
    }
    for name, model in expected_models.items():
        assert agents[name]["primary"] == model
        assert agents[name]["fallback"] == []

    affinity = data["policy"]["strict_provider_affinity"]
    expected_affinity = {
        "friday": "claude-cli",
        "arsenal": "codex-cli",
        "jocasta": "nvidia",
        "edith": "google-gemini-cli",
    }
    for name, provider in expected_affinity.items():
        assert affinity[name] == provider


def test_role_profiles_match_canonical_lanes() -> None:
    profiles = Path("templates/profiles")
    expected_models = {
        "friday": ("claude-cli/claude-opus-4-6", "claude-cli"),
        "arsenal": ("codex-cli/gpt-5.3-codex", "codex-cli"),
        "jocasta": ("nvidia/moonshotai/kimi-k2.5", "nvidia"),
        "edith": ("google-gemini-cli/gemini-3-pro-preview", "google-gemini-cli"),
    }
    for name, (model, provider) in expected_models.items():
        profile = yaml.safe_load((profiles / f"{name}.profile.json").read_text(encoding="utf-8"))
        assert profile["model"]["primary"] == model
        assert profile["model"]["fallbacks"] == []
        assert profile["model"]["providerLock"] == provider


def test_single_task_profile_is_solo_and_opus_only() -> None:
    profile = yaml.safe_load(
        Path("templates/profiles/single-task-role.profile.json").read_text(encoding="utf-8"),
    )
    assert profile["model"]["primary"] == "anthropic/claude-opus-4-6"
    assert profile["model"]["fallbacks"] == []
    assert profile["model"]["providerLock"] == "anthropic"
    assert profile["persona"]["mode"] == "individual"
    assert profile["persona"]["orchestratorEnabled"] is False
