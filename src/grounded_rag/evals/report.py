import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grounded_rag.evals.ablations import Ablation
from grounded_rag.evals.judge import JUDGE_PROMPT_VERSION
from grounded_rag.evals.metrics import (
    Proportion,
    abstention_scores,
    mean,
    percentile,
    proportion,
)
from grounded_rag.evals.runner import RETRIEVAL_KS, QuestionRecord, RunOutput, SecurityRecord
from grounded_rag.llm.prompts import PROMPT_VERSION

SUMMARY_START = "<!-- summary:start -->"
SUMMARY_END = "<!-- summary:end -->"
_HEADLINE_K = 5


def _rate(p: Proportion) -> dict[str, Any]:
    return asdict(p)


def _share(records: Sequence[QuestionRecord], predicate: Any) -> Proportion:
    return proportion(sum(1 for r in records if predicate(r)), len(records))


def summarize_config(ablation: Ablation, records: Sequence[QuestionRecord]) -> dict[str, Any]:
    answerable = [r for r in records if r.answerable]
    generated = [r for r in records if r.first_pass_verified_rate is not None]
    answered = [r for r in records if not r.abstained]
    scores = abstention_scores([(r.answerable, r.abstained) for r in records])
    with_retrieval = [r for r in answerable if r.hit_at]

    def verdicts(label: str) -> dict[str, Any]:
        return _rate(_share(answerable, lambda r: r.correctness == label))

    by_category: dict[str, list[QuestionRecord]] = defaultdict(list)
    for record in answerable:
        by_category[record.category].append(record)

    return {
        "config": ablation.describe(),
        "questions": len(records),
        "answerable": len(answerable),
        "unanswerable": len(records) - len(answerable),
        "retrieval": {
            "hit_at": {
                str(k): _rate(_share(with_retrieval, lambda r, k=k: r.hit_at.get(k, False)))
                for k in RETRIEVAL_KS
            },
            "recall_at": {
                str(k): mean([r.recall_at.get(k, 0.0) for r in with_retrieval])
                for k in RETRIEVAL_KS
            },
            "mrr": mean([r.reciprocal_rank or 0.0 for r in with_retrieval]),
            "context_hit": _rate(_share(with_retrieval, lambda r: bool(r.context_hit))),
        },
        "answers": {
            "correct": verdicts("correct"),
            "partially_correct": verdicts("partially_correct"),
            "incorrect": verdicts("incorrect"),
            "abstention_precision": _rate(scores.precision),
            "abstention_recall": _rate(scores.recall),
            "false_abstention": _rate(scores.false_abstention),
        },
        "faithfulness": {
            "mean_first_pass_citation_verification": mean(
                [r.first_pass_verified_rate or 0.0 for r in generated]
            ),
            "all_citations_verified_first_pass": _rate(
                _share(generated, lambda r: r.first_pass_verified_rate == 1.0)
            ),
            "needed_repair": _rate(_share(generated, lambda r: r.repaired)),
            "judge_supported": _rate(_share(answered, lambda r: r.supported is True)),
        },
        "operations": {
            "latency_ms_p50": percentile([r.latency_ms for r in records], 50),
            "latency_ms_p95": percentile([r.latency_ms for r in records], 95),
            "tokens_per_query": mean(
                [r.prompt_tokens + r.completion_tokens + r.embedding_tokens for r in records]
            ),
            "cost_usd_per_query": mean([r.cost_usd for r in records]),
            "model_calls_per_query": mean([float(r.llm_calls) for r in records]),
            "rewrites_per_query": mean([float(r.rewrites) for r in records]),
        },
        "by_category": {
            category: {
                "answerable": len(items),
                "correct": _rate(_share(items, lambda r: r.correctness == "correct")),
                "hit_at_5": _rate(
                    _share([r for r in items if r.hit_at], lambda r: r.hit_at.get(5, False))
                ),
            }
            for category, items in sorted(by_category.items())
        },
    }


def summarize_security(
    security: Sequence[SecurityRecord], ablations: Sequence[Ablation]
) -> dict[str, Any]:
    by_config = {
        a.name: {
            "probes": sum(1 for s in security if s.config == a.name),
            "leaks": sum(1 for s in security if s.config == a.name and s.leaked),
        }
        for a in ablations
    }
    return {
        "probes": len(security),
        "leaks": sum(1 for s in security if s.leaked),
        "by_config": by_config,
        "findings": [
            {"config": s.config, "probe": s.probe_id, "findings": s.findings}
            for s in security
            if s.leaked
        ],
    }


def build_report(
    output: RunOutput, ablations: Sequence[Ablation], meta: Mapping[str, Any]
) -> dict[str, Any]:
    configs = [
        summarize_config(a, [r for r in output.records if r.config == a.name]) for a in ablations
    ]
    return {
        "meta": {**meta, "split": output.split, "questions": output.questions},
        "configs": configs,
        "security": summarize_security(output.security, ablations),
    }


def _pct(p: Mapping[str, Any]) -> str:
    if p["value"] is None:
        return "n/a"
    return f"{p['value'] * 100:.1f}% [{p['low'] * 100:.1f}, {p['high'] * 100:.1f}]"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _seconds(value_ms: float | None) -> str:
    return "n/a" if value_ms is None else f"{value_ms / 1000:.2f}"


def summary_table(report: Mapping[str, Any]) -> str:
    rows = [
        "| Configuration | hit@5 | MRR | Correct | Abstention precision | Abstention recall "
        "| Leaks | p50 s | p95 s | USD/query |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    security = report["security"]["by_config"]
    for config in report["configs"]:
        name = config["config"]["name"]
        leaks = security.get(name, {"probes": 0, "leaks": 0})
        answers, ops = config["answers"], config["operations"]
        rows.append(
            f"| {name} | {_pct(config['retrieval']['hit_at'][str(_HEADLINE_K)])} "
            f"| {_num(config['retrieval']['mrr'])} | {_pct(answers['correct'])} "
            f"| {_pct(answers['abstention_precision'])} | {_pct(answers['abstention_recall'])} "
            f"| {leaks['leaks']} / {leaks['probes']} | {_seconds(ops['latency_ms_p50'])} "
            f"| {_seconds(ops['latency_ms_p95'])} | {_num(ops['cost_usd_per_query'], 5)} |"
        )
    meta = report["meta"]
    note = (
        f"Split: {meta['split']} ({meta['questions']} questions). Answers by "
        f"`{meta['chat_model']}`, judged by `{meta['judge_model']}` "
        f"(prompt {meta['judge_prompt_version']}). Intervals are 95% Wilson. "
        f"Run `{meta['run_id']}`."
    )
    return "\n".join([SUMMARY_START, *rows, "", note, SUMMARY_END])


def render_markdown(report: Mapping[str, Any]) -> str:
    meta = report["meta"]
    lines = [
        f"# Evaluation report: {meta['run_id']} ({meta['split']} split)",
        "",
        f"- Answer model: `{meta['chat_model']}`; embedding model: `{meta['embedding_model']}`",
        f"- Judge model: `{meta['judge_model']}`, prompt version `{meta['judge_prompt_version']}`; "
        f"answer prompts `{meta['prompt_version']}`",
        f"- Questions: {meta['questions']} ({meta['concurrency']} at a time). "
        f"Repository revision: `{meta['git_sha']}`",
        f"- Provider cache: {meta['cache']['hits']} replayed, {meta['cache']['misses']} live. "
        "Costs are list-price figures computed from token counts, so replays report what a live "
        "run would cost; latency adds back the time the original calls took.",
        "",
        "Rates show the value and a 95% Wilson interval. Retrieval figures use answerable "
        "questions and the ranking of the first retrieval.",
        "",
        "## Summary",
        "",
        summary_table(report),
        "",
    ]
    configs = report["configs"]

    def table(title: str, header: list[str], rows: list[list[str]]) -> None:
        lines.extend([f"## {title}", "", "| " + " | ".join(header) + " |"])
        lines.append("|" + "---|" * len(header))
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        lines.append("")

    ks = [str(k) for k in RETRIEVAL_KS]
    table(
        "Retrieval",
        [
            "Configuration",
            *[f"hit@{k}" for k in ks],
            f"recall@{ks[-1]}",
            "MRR",
            "Final context hit",
        ],
        [
            [
                c["config"]["name"],
                *[_pct(c["retrieval"]["hit_at"][k]) for k in ks],
                _num(c["retrieval"]["recall_at"][ks[-1]]),
                _num(c["retrieval"]["mrr"]),
                _pct(c["retrieval"]["context_hit"]),
            ]
            for c in configs
        ],
    )
    table(
        "Answers",
        [
            "Configuration",
            "Correct",
            "Partially correct",
            "Incorrect",
            "False abstention",
            "Abstention precision",
            "Abstention recall",
        ],
        [
            [
                c["config"]["name"],
                _pct(c["answers"]["correct"]),
                _pct(c["answers"]["partially_correct"]),
                _pct(c["answers"]["incorrect"]),
                _pct(c["answers"]["false_abstention"]),
                _pct(c["answers"]["abstention_precision"]),
                _pct(c["answers"]["abstention_recall"]),
            ]
            for c in configs
        ],
    )
    table(
        "Faithfulness",
        [
            "Configuration",
            "Mean citation verification, first pass",
            "All citations verified, first pass",
            "Needed repair",
            "Judge: supported by quotes",
        ],
        [
            [
                c["config"]["name"],
                _num(c["faithfulness"]["mean_first_pass_citation_verification"]),
                _pct(c["faithfulness"]["all_citations_verified_first_pass"]),
                _pct(c["faithfulness"]["needed_repair"]),
                _pct(c["faithfulness"]["judge_supported"]),
            ]
            for c in configs
        ],
    )
    table(
        "Operations",
        ["Configuration", "p50 s", "p95 s", "Tokens/query", "USD/query", "Model calls/query"],
        [
            [
                c["config"]["name"],
                _seconds(c["operations"]["latency_ms_p50"]),
                _seconds(c["operations"]["latency_ms_p95"]),
                _num(c["operations"]["tokens_per_query"], 0),
                _num(c["operations"]["cost_usd_per_query"], 5),
                _num(c["operations"]["model_calls_per_query"], 2),
            ]
            for c in configs
        ],
    )
    security = report["security"]
    lines.extend(
        [
            "## Security",
            "",
            f"{security['probes']} probes ran across all configurations, "
            f"{security['leaks']} leaked (this must be 0).",
            "",
            "| Configuration | Probes | Leaks |",
            "|---|---|---|",
            *[
                f"| {name} | {item['probes']} | {item['leaks']} |"
                for name, item in security["by_config"].items()
            ],
            "",
        ]
    )
    for finding in security["findings"]:
        lines.append(f"- LEAK in {finding['config']}, {finding['probe']}: {finding['findings']}")
    categories = sorted({cat for c in configs for cat in c["by_category"]})
    if categories and configs:
        lines.extend(
            ["## By category (answerable questions, share judged correct)", ""]
            + [f"| Category | {' | '.join(c['config']['name'] for c in configs)} |"]
            + ["|" + "---|" * (len(configs) + 1)]
            + [
                f"| {cat} | "
                + " | ".join(
                    _pct(c["by_category"][cat]["correct"]) if cat in c["by_category"] else "n/a"
                    for c in configs
                )
                + " |"
                for cat in categories
            ]
            + [""]
        )
    return "\n".join(lines)


def git_short_sha(cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    return completed.stdout.strip() or "nogit"


def run_directory(root: Path, repository: Path, today: datetime | None = None) -> Path:
    stamp = (today or datetime.now(UTC)).strftime("%Y-%m-%d")
    return root / f"{stamp}-{git_short_sha(repository)}"


def report_meta(
    *,
    run_id: str,
    git_sha: str,
    chat_model: str,
    judge_model: str,
    embedding_model: str,
    concurrency: int,
    cache_hits: int,
    cache_misses: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "git_sha": git_sha,
        "chat_model": chat_model,
        "judge_model": judge_model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "embedding_model": embedding_model,
        "concurrency": concurrency,
        "cache": {"hits": cache_hits, "misses": cache_misses},
    }


def write_report(
    directory: Path, report: Mapping[str, Any], records: Sequence[QuestionRecord]
) -> list[Path]:
    split = report["meta"]["split"]
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": directory / f"report-{split}.json",
        "md": directory / f"report-{split}.md",
        "records": directory / f"records-{split}.jsonl",
    }
    paths["json"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    paths["md"].write_text(render_markdown(report) + "\n", encoding="utf-8", newline="\n")
    paths["records"].write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    return list(paths.values())
