from pathlib import Path

import yaml


def test_bot_runtime_model_policy_is_opus_only() -> None:
    data = yaml.safe_load(Path("config/model-policy.public.yaml").read_text(encoding="utf-8"))
    agents = data["agents"]
    for name in ("friday", "arsenal", "jocasta", "edith"):
        assert agents[name]["primary"] == "anthropic/claude-opus-4-6"
        assert agents[name]["fallback"] == []

    affinity = data["policy"]["strict_provider_affinity"]
    for name in ("friday", "arsenal", "jocasta", "edith"):
        assert affinity[name] == "anthropic"


def test_role_profiles_are_opus_only() -> None:
    profiles = Path("templates/profiles")
    for name in ("friday", "arsenal", "jocasta", "edith"):
        profile = yaml.safe_load((profiles / f"{name}.profile.json").read_text(encoding="utf-8"))
        assert profile["model"]["primary"] == "anthropic/claude-opus-4-6"
        assert profile["model"]["fallbacks"] == []
        assert profile["model"]["providerLock"] == "anthropic"


def test_single_task_profile_is_solo_and_opus_only() -> None:
    profile = yaml.safe_load(
        Path("templates/profiles/single-task-role.profile.json").read_text(encoding="utf-8"),
    )
    assert profile["model"]["primary"] == "anthropic/claude-opus-4-6"
    assert profile["model"]["fallbacks"] == []
    assert profile["model"]["providerLock"] == "anthropic"
    assert profile["persona"]["mode"] == "individual"
    assert profile["persona"]["orchestratorEnabled"] is False
