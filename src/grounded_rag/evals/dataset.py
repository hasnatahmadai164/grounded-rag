import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from grounded_rag.domain.principal import Principal
from grounded_rag.errors import ConfigurationError

ACCESS_LABELS = ("legal", "finance", "operations")
ADMIN_ROLE = "admin"
DEV_CONTRACTS = 10

# One template per CUAD clause category. Each asks for what the contract says, so a contract
# that lacks the clause has no answer and the correct behaviour is to abstain.
QUESTION_TEMPLATES: dict[str, str] = {
    "Governing Law": 'Which law governs the "{title}"?',
    "Expiration Date": 'When does the initial term of the "{title}" expire?',
    "Renewal Term": 'What is the renewal term of the "{title}"?',
    "Notice Period To Terminate Renewal": (
        'What notice must be given to stop the "{title}" from renewing?'
    ),
    "Anti-Assignment": 'What restrictions does the "{title}" place on assignment of the agreement?',
    "Non-Compete": 'What non-compete restrictions apply under the "{title}"?',
    "Exclusivity": 'What exclusivity commitments are in the "{title}"?',
    "Cap On Liability": 'What cap on liability applies under the "{title}"?',
    "Audit Rights": 'What audit rights are granted under the "{title}"?',
    "Termination For Convenience": (
        'Can a party terminate the "{title}" for convenience, and on what terms?'
    ),
    "Insurance": 'What insurance must be maintained under the "{title}"?',
    "License Grant": 'What license is granted under the "{title}"?',
    "Change Of Control": 'What happens on a change of control under the "{title}"?',
    "Minimum Commitment": 'What minimum purchase or volume commitment is in the "{title}"?',
}

Split = Literal["dev", "test"]


class GoldSpan(BaseModel):
    start: int
    end: int
    text: str


class EvalContract(BaseModel):
    id: str
    title: str
    display_title: str
    file: str
    sha256: str
    access_label: str
    split: Split

    @property
    def allowed_roles(self) -> frozenset[str]:
        return frozenset({self.access_label, ADMIN_ROLE})


class EvalQuestion(BaseModel):
    id: str
    contract_id: str
    category: str
    question: str
    answerable: bool
    gold_spans: list[GoldSpan]
    split: Split


class EvalDataset(BaseModel):
    source: dict[str, Any]
    contracts: list[EvalContract]
    questions: list[EvalQuestion]

    def questions_in(self, split: Split) -> list[EvalQuestion]:
        return [q for q in self.questions if q.split == split]

    def contract(self, contract_id: str) -> EvalContract:
        return next(c for c in self.contracts if c.id == contract_id)


def principal_for(label: str) -> Principal:
    return Principal(name=f"eval-{label}", roles=frozenset({label}))


ADMIN_PRINCIPAL = Principal(name="eval-admin", roles=frozenset({ADMIN_ROLE}))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def display_title(title: str) -> str:
    """A readable name for a CUAD file title, which is how questions refer to the contract."""
    name = re.sub(r"\.(pdf|txt)$", "", title, flags=re.IGNORECASE)
    return " ".join(re.sub(r"[_\-]+", " ", name).split())


def access_label(contract_id: str) -> str:
    return ACCESS_LABELS[int(_digest(f"label:{contract_id}")[:8], 16) % len(ACCESS_LABELS)]


def assign_splits(contract_ids: list[str], dev_count: int) -> dict[str, Split]:
    """Order contracts by a hash so the dev set is fixed and independent of listing order."""
    ordered = sorted(contract_ids, key=lambda cid: _digest(f"split:{cid}"))
    return {cid: ("dev" if rank < dev_count else "test") for rank, cid in enumerate(ordered)}


def build_dataset(
    subset: dict[str, Any], contracts_dir: Path, dev_count: int = DEV_CONTRACTS
) -> EvalDataset:
    """Turn the annotations written by `scripts/fetch_cuad.py` into evaluation questions."""
    ids = {entry["title"]: Path(entry["file"]).stem for entry in subset["contracts"]}
    splits = assign_splits(list(ids.values()), dev_count)
    contracts: list[EvalContract] = []
    questions: list[EvalQuestion] = []
    for entry in subset["contracts"]:
        contract_id = ids[entry["title"]]
        text = (contracts_dir / entry["file"]).read_text(encoding="utf-8")
        contract = EvalContract(
            id=contract_id,
            title=entry["title"],
            display_title=display_title(entry["title"]),
            file=entry["file"],
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            access_label=access_label(contract_id),
            split=splits[contract_id],
        )
        contracts.append(contract)
        for item in entry["questions"]:
            category = item["category"]
            if category not in QUESTION_TEMPLATES:
                continue
            questions.append(
                EvalQuestion(
                    id=f"{contract_id}:{re.sub(r'[^a-z0-9]+', '-', category.lower()).strip('-')}",
                    contract_id=contract_id,
                    category=category,
                    question=QUESTION_TEMPLATES[category].format(title=contract.display_title),
                    answerable=bool(item["answerable"]),
                    gold_spans=[GoldSpan(**span) for span in item["answers"]],
                    split=contract.split,
                )
            )
    source = dict(subset["source"])
    source["categories"] = sorted(QUESTION_TEMPLATES)
    return EvalDataset(source=source, contracts=contracts, questions=questions)


def write_dataset(dataset: EvalDataset, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"source": dataset.source, "contracts": [c.model_dump() for c in dataset.contracts]}
    (directory / "contracts.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    for split in ("dev", "test"):
        lines = [q.model_dump_json() for q in dataset.questions_in(split)]
        (directory / f"{split}.jsonl").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n"
        )


def load_dataset(directory: Path) -> EvalDataset:
    manifest_path = directory / "contracts.json"
    if not manifest_path.exists():
        raise ConfigurationError(
            f"No dataset in {directory}. Run `grounded-rag eval build-dataset` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    questions: list[EvalQuestion] = []
    for split in ("dev", "test"):
        path = directory / f"{split}.jsonl"
        if path.exists():
            questions.extend(
                EvalQuestion.model_validate_json(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    return EvalDataset(
        source=manifest["source"],
        contracts=[EvalContract.model_validate(c) for c in manifest["contracts"]],
        questions=questions,
    )
