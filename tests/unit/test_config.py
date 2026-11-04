from pathlib import Path

import pytest

from grounded_rag.config import Settings

ROOT = Path(__file__).resolve().parents[2]


def documented_names() -> set[str]:
    names = set()
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            names.add(line.split("=", 1)[0].strip().lower())
    return names


def test_every_setting_is_documented_in_env_example() -> None:
    assert set(Settings.model_fields) == documented_names()


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_POOL_SIZE", "11")
    assert Settings(_env_file=None).database_pool_size == 11


def test_empty_environment_value_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "")
    assert Settings(_env_file=None).log_level == "INFO"


def test_content_logging_is_off_by_default() -> None:
    assert Settings(_env_file=None).log_content is False


def test_every_setting_appears_in_the_readme_configuration_table() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = [name for name in Settings.model_fields if f"| `{name.upper()}` |" not in readme]
    assert missing == []
