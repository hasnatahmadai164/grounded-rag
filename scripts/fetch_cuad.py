"""Download CUAD and extract the contract subset used by the evaluation harness.

Standard library only. Writes plain-text contracts and a `subset.json` holding the
clause annotations for the selected categories. The raw dataset is never committed.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

CUAD_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
LICENSE = "CC BY 4.0"
CITATION = (
    "Hendrycks, D., Burns, C., Chen, A., and Ball, S. CUAD: An Expert-Annotated NLP Dataset "
    "for Legal Contract Review. NeurIPS 2021 Datasets and Benchmarks."
)
BYTE_ORDER_MARK = chr(0xFEFF)
CATEGORIES = [
    "Governing Law",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Anti-Assignment",
    "Non-Compete",
    "Exclusivity",
    "Cap On Liability",
    "Audit Rights",
    "Termination For Convenience",
    "Insurance",
    "License Grant",
    "Change Of Control",
    "Minimum Commitment",
]


def normalize(text: str) -> str:
    """Same text normalisation as the service's document loader."""
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\x00", "")
        .replace(BYTE_ORDER_MARK, "")
    )


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(destination)


def load_cuad(zip_path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as archive:
        member = next((n for n in archive.namelist() if n.endswith("CUAD_v1.json")), None)
        if member is None:
            raise SystemExit(f"{zip_path} does not contain CUAD_v1.json")
        with archive.open(member) as handle:
            return json.load(handle)["data"]  # type: ignore[no-any-return]


def category_of(qa: dict[str, Any]) -> str | None:
    identifier = str(qa.get("id", ""))
    if "__" in identifier:
        return identifier.rsplit("__", 1)[1]
    match = re.search(r'related to "([^"]+)"', str(qa.get("question", "")))
    return match.group(1) if match else None


def slugify(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:80] or "contract"
    return f"{slug}_{hashlib.sha256(title.encode()).hexdigest()[:8]}"


def select_contracts(
    data: list[dict[str, Any]], count: int, seed: int, min_chars: int, max_chars: int
) -> list[dict[str, Any]]:
    """Pick contracts of moderate length, ordered by a seeded hash so the choice is repeatable."""
    eligible = [
        entry for entry in data if min_chars <= len(entry["paragraphs"][0]["context"]) <= max_chars
    ]
    eligible.sort(key=lambda e: hashlib.sha256(f"{seed}:{e['title']}".encode()).hexdigest())
    return eligible[:count]


def relocate_answer(context: str, normalized: str, answer: dict[str, Any]) -> dict[str, Any] | None:
    """Express an answer span in the normalised text, or None when it cannot be located."""
    text = normalize(str(answer["text"]))
    if not text:
        return None
    start = len(normalize(context[: int(answer["answer_start"])]))
    if normalized[start : start + len(text)] != text:
        start = normalized.find(text)
        if start < 0:
            return None
    return {"text": text, "start": start, "end": start + len(text)}


def build_subset(
    entries: list[dict[str, Any]], categories: list[str], output: Path
) -> dict[str, Any]:
    wanted = {name.lower(): name for name in categories}
    contracts_dir = output / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    contracts: list[dict[str, Any]] = []
    dropped = 0
    for entry in entries:
        paragraph = entry["paragraphs"][0]
        context = paragraph["context"]
        normalized = normalize(context)
        file_name = f"{slugify(entry['title'])}.txt"
        (contracts_dir / file_name).write_text(normalized, encoding="utf-8", newline="\n")
        questions = []
        for qa in paragraph["qas"]:
            category = wanted.get((category_of(qa) or "").lower())
            if category is None:
                continue
            impossible = bool(qa.get("is_impossible", False))
            spans = []
            for answer in qa.get("answers", []):
                span = relocate_answer(context, normalized, answer)
                if span is None:
                    dropped += 1
                else:
                    spans.append(span)
            if not impossible and not spans:
                # Annotated as present but unlocatable: keeping it would mislabel it as absent.
                continue
            questions.append({"category": category, "answerable": not impossible, "answers": spans})
        contracts.append(
            {"title": entry["title"], "file": f"contracts/{file_name}", "questions": questions}
        )
    return {
        "source": {"url": CUAD_URL, "license": LICENSE, "citation": CITATION},
        "categories": categories,
        "dropped_answers": dropped,
        "contracts": contracts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/cuad"))
    parser.add_argument("--zip", type=Path, help="Use an already downloaded CUAD_v1.zip.")
    parser.add_argument("--contracts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-chars", type=int, default=8_000)
    parser.add_argument("--max-chars", type=int, default=120_000)
    args = parser.parse_args(argv)

    zip_path = args.zip or args.output / "CUAD_v1.zip"
    if not zip_path.exists():
        print(f"Downloading {CUAD_URL}", file=sys.stderr)
        download(CUAD_URL, zip_path)
    data = load_cuad(zip_path)
    selected = select_contracts(data, args.contracts, args.seed, args.min_chars, args.max_chars)
    subset = build_subset(selected, CATEGORIES, args.output)
    (args.output / "subset.json").write_text(
        json.dumps(subset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"Wrote {len(selected)} contracts to {args.output} "
        f"({subset['dropped_answers']} answer spans could not be located and were dropped)."
    )
    print(f"Dataset license: {LICENSE}. Keep the attribution in docs/evaluation.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
