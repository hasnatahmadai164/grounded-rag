"""Copy the latest evaluation summary table into the README.

Standard library only. The README marks the spot with `<!-- results:start -->` and
`<!-- results:end -->`; everything between them is replaced.
"""

import argparse
import sys
from pathlib import Path

README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
SUMMARY_START = "<!-- summary:start -->"
SUMMARY_END = "<!-- summary:end -->"


def latest_report(reports: Path, split: str) -> Path | None:
    """The newest `report-<split>.md`. Run directories start with an ISO date, so names sort."""
    candidates = (
        sorted(
            (d for d in reports.iterdir() if d.is_dir() and (d / f"report-{split}.md").exists()),
            key=lambda d: d.name,
            reverse=True,
        )
        if reports.is_dir()
        else []
    )
    return candidates[0] / f"report-{split}.md" if candidates else None


def extract_summary(report_text: str) -> str:
    start = report_text.find(SUMMARY_START)
    end = report_text.find(SUMMARY_END)
    if start < 0 or end < start:
        raise ValueError("The report has no summary block.")
    return report_text[start + len(SUMMARY_START) : end].strip()


def inject(readme_text: str, summary: str, report_link: str) -> str:
    start = readme_text.find(README_START)
    end = readme_text.find(README_END)
    if start < 0 or end < start:
        raise ValueError(f"The README needs {README_START} and {README_END} markers.")
    block = f"{summary}\n\nFull report: [{report_link}]({report_link})"
    return f"{readme_text[: start + len(README_START)]}\n{block}\n{readme_text[end:]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("evals/reports"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--check", action="store_true", help="Fail if the README is out of date.")
    args = parser.parse_args(argv)

    report = latest_report(args.reports, args.split)
    if report is None:
        print(f"No {args.split} report found under {args.reports}.", file=sys.stderr)
        return 1
    try:
        summary = extract_summary(report.read_text(encoding="utf-8"))
        current = args.readme.read_text(encoding="utf-8")
        updated = inject(current, summary, report.as_posix())
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    if updated == current:
        print("README is already up to date.")
        return 0
    if args.check:
        print("README is out of date. Run scripts/inject_results.py.", file=sys.stderr)
        return 1
    args.readme.write_text(updated, encoding="utf-8", newline="\n")
    print(f"Updated {args.readme} from {report}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
