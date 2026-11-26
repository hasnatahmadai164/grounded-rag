import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "inject_results.py"
TABLE = "| Configuration | hit@5 |\n|---|---|\n| hybrid | 80.0% |"


@pytest.fixture(scope="module")
def inject() -> ModuleType:
    spec = importlib.util.spec_from_file_location("inject_results", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_report(root: Path, run: str, split: str = "test", table: str = TABLE) -> Path:
    directory = root / run
    directory.mkdir(parents=True)
    path = directory / f"report-{split}.md"
    path.write_text(
        f"# Report\n\n<!-- summary:start -->\n{table}\n\nNote.\n<!-- summary:end -->\n\n## More\n",
        encoding="utf-8",
    )
    return path


README = "# Project\n\n## Results\n\n<!-- results:start -->\nold\n<!-- results:end -->\n\n## Next\n"


def test_the_table_replaces_whatever_sat_between_the_markers(
    inject: ModuleType, tmp_path: Path
) -> None:
    make_report(tmp_path / "reports", "2026-01-01-aaaaaaa")
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")

    code = inject.main(["--reports", str(tmp_path / "reports"), "--readme", str(readme)])

    text = readme.read_text(encoding="utf-8")
    assert code == 0
    assert "old" not in text
    assert TABLE in text
    assert "Full report:" in text
    assert text.startswith("# Project") and text.endswith("## Next\n")


def test_running_twice_changes_nothing_the_second_time(inject: ModuleType, tmp_path: Path) -> None:
    make_report(tmp_path / "reports", "2026-01-01-aaaaaaa")
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")
    args = ["--reports", str(tmp_path / "reports"), "--readme", str(readme)]
    inject.main(args)
    first = readme.read_text(encoding="utf-8")

    assert inject.main(args) == 0
    assert readme.read_text(encoding="utf-8") == first


def test_the_newest_run_for_the_requested_split_is_used(inject: ModuleType, tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    make_report(reports, "2026-01-01-aaaaaaa", table="| old |")
    make_report(reports, "2026-02-01-bbbbbbb", table="| new |")
    make_report(reports, "2026-03-01-ccccccc", split="dev", table="| dev |")

    latest = inject.latest_report(reports, "test")

    assert latest == reports / "2026-02-01-bbbbbbb" / "report-test.md"


def test_check_mode_fails_when_the_readme_is_stale_and_does_not_write(
    inject: ModuleType, tmp_path: Path
) -> None:
    make_report(tmp_path / "reports", "2026-01-01-aaaaaaa")
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")

    code = inject.main(["--reports", str(tmp_path / "reports"), "--readme", str(readme), "--check"])

    assert code == 1
    assert readme.read_text(encoding="utf-8") == README


def test_a_readme_without_markers_is_rejected(inject: ModuleType, tmp_path: Path) -> None:
    make_report(tmp_path / "reports", "2026-01-01-aaaaaaa")
    readme = tmp_path / "README.md"
    readme.write_text("# No markers\n", encoding="utf-8")

    code = inject.main(["--reports", str(tmp_path / "reports"), "--readme", str(readme)])

    assert code == 2
    assert readme.read_text(encoding="utf-8") == "# No markers\n"


def test_no_report_yet_is_reported_clearly(inject: ModuleType, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")

    assert inject.main(["--reports", str(tmp_path / "none"), "--readme", str(readme)]) == 1
