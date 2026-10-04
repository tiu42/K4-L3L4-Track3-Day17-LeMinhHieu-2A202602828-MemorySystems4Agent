from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent, message_text
from config import load_config
from memory_store import estimate_tokens, normalize_text
from model_provider import build_chat_model

HEADERS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]

_REFUSAL = re.compile(r"chưa có thông tin|không có bộ nhớ|chỉ dựa được vào nội dung", re.I)


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int
    details: list[dict[str, Any]] = field(default_factory=list)


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _contains(answer: str, needle: str) -> bool:
    return normalize_text(needle).casefold() in normalize_text(answer).casefold()


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if only some do, 0 if none."""

    if not expected:
        return 1.0
    hits = sum(_contains(answer, item) for item in expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight offline quality score in [0, 1].

    - 0.6 x fraction of expected facts covered (correctness)
    - 0.2 if the agent actually answers instead of saying it does not know
    - 0.2 if the answer is concise (<= 80 estimated tokens), matching the user's style
    """

    coverage = sum(_contains(answer, item) for item in expected) / len(expected) if expected else 1.0
    answered = 0.0 if not answer.strip() or _REFUSAL.search(answer) else 1.0
    concise = 1.0 if 0 < estimate_tokens(answer) <= 80 else 0.0
    return round(0.6 * coverage + 0.2 * answered + 0.2 * concise, 4)


def llm_judge_quality(judge, question: str, answer: str, expected: list[str]) -> float | None:
    """Live-mode judge: ask the judge model for a 0-10 score. Returns None on failure."""

    prompt = (
        "Chấm điểm câu trả lời của trợ lý từ 0 đến 10 theo: đúng fact mong đợi, không dùng fact cũ đã bị đính chính, "
        "ngắn gọn. Chỉ trả về một số.\n"
        f"Câu hỏi: {question}\nFact mong đợi: {', '.join(expected)}\nCâu trả lời: {answer}"
    )
    try:
        match = re.search(r"\d+(?:\.\d+)?", message_text(judge.invoke(prompt)))
        return min(10.0, float(match.group())) / 10 if match else None
    except Exception:
        return None


def run_agent_benchmark(
    agent_name: str, agent, conversations: list[dict[str, Any]], config, judge=None
) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    Each conversation runs in its own thread; every recall question is then asked
    in a *fresh* thread, so only persistent memory can answer it.
    """

    user_ids = sorted({conv["user_id"] for conv in conversations})
    initial_sizes = {user: agent.memory_file_size(user) for user in user_ids}
    threads: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    details: list[dict[str, Any]] = []

    for conv in conversations:
        thread_id = f"{agent_name}-{conv['id']}"
        threads.append(thread_id)
        for turn in conv["turns"]:
            agent.reply(conv["user_id"], thread_id, turn)

        for index, item in enumerate(conv.get("recall_questions", []), start=1):
            recall_thread = f"{thread_id}-recall-{index}"
            threads.append(recall_thread)
            answer = agent.reply(conv["user_id"], recall_thread, item["question"])["response"]
            expected = item["expected_contains"]
            recall = recall_points(answer, expected)
            quality = heuristic_quality(answer, expected)
            if judge is not None:
                judged = llm_judge_quality(judge, item["question"], answer, expected)
                quality = judged if judged is not None else quality
            recall_scores.append(recall)
            quality_scores.append(quality)
            details.append(
                {"conversation": conv["id"], "question": item["question"], "answer": answer, "recall": recall}
            )

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(t) for t in threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(t) for t in threads),
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=sum(agent.memory_file_size(u) - initial_sizes[u] for u in user_ids),
        compactions=sum(agent.compaction_count(t) for t in threads),
        details=details,
    )


def _table_values(rows: list[BenchmarkRow]) -> list[list[Any]]:
    return [
        [
            row.agent_name,
            f"{row.agent_tokens_only:,}",
            f"{row.prompt_tokens_processed:,}",
            f"{row.recall_score:.0%}",
            f"{row.response_quality:.2f}",
            f"{row.memory_growth_bytes:,}",
            row.compactions,
        ]
        for row in rows
    ]


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render rows as a GitHub-flavored markdown table."""

    values = _table_values(rows)
    try:
        from tabulate import tabulate

        return tabulate(values, headers=HEADERS, tablefmt="github", disable_numparse=True, colalign=("left",) + ("right",) * 6)
    except ImportError:
        lines = ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in values]
        return "\n".join(lines)


def _delta(base: int, adv: int) -> str:
    if not base:
        return "n/a"
    return f"{(adv - base) / base:+.1%}"


def compare_rows(baseline: BenchmarkRow, advanced: BenchmarkRow) -> str:
    return (
        f"Advanced vs Baseline: agent tokens {_delta(baseline.agent_tokens_only, advanced.agent_tokens_only)}, "
        f"prompt tokens {_delta(baseline.prompt_tokens_processed, advanced.prompt_tokens_processed)}, "
        f"recall {advanced.recall_score - baseline.recall_score:+.0%}"
    )


def run_suite(title: str, dataset: Path, config, force_offline: bool, judge=None) -> tuple[str, list[BenchmarkRow]]:
    conversations = load_conversations(dataset)
    baseline = BaselineAgent(config, force_offline=force_offline)
    advanced = AdvancedAgent(config, force_offline=force_offline)
    # Start every suite from an empty User.md so memory growth is measured from zero.
    for user_id in {conv["user_id"] for conv in conversations}:
        advanced.profile_store.delete(user_id)

    rows = [
        run_agent_benchmark("Baseline", baseline, conversations, config, judge),
        run_agent_benchmark("Advanced", advanced, conversations, config, judge),
    ]
    turns = sum(len(c["turns"]) for c in conversations)
    questions = sum(len(c.get("recall_questions", [])) for c in conversations)
    header = (
        f"## {title}\n\nDataset: `{dataset.relative_to(config.base_dir).as_posix()}` "
        f"({len(conversations)} conversations, {turns} turns, {questions} recall questions, "
        f"mode: {advanced.mode})\n"
    )
    body = f"{format_rows(rows)}\n\n{compare_rows(*rows)}\n"
    return header + "\n" + body, rows


def _format_details(rows: list[BenchmarkRow]) -> str:
    lines = []
    for row in rows:
        lines.append(f"### {row.agent_name}")
        for item in row.details:
            answer = item["answer"].replace("\n", " / ")
            lines.append(f"- [{item['conversation']}] recall={item['recall']:.1f} | Q: {item['question']} | A: {answer}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the Standard benchmark and the Long-Context Stress benchmark."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Day 17 memory benchmark: Baseline vs Advanced")
    parser.add_argument("--live", action="store_true", help="use the live LLM (needs LAB_MODE=live + API key)")
    parser.add_argument("--verbose", action="store_true", help="print every recall question and answer")
    parser.add_argument("--output", type=Path, help="also write the report to this markdown file")
    args = parser.parse_args()

    config = load_config(Path(__file__).resolve().parent.parent)
    force_offline = not args.live
    judge = build_chat_model(config.judge_model) if args.live and config.live_mode else None

    suites = [
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    ]
    sections = []
    for title, dataset in suites:
        text, rows = run_suite(title, dataset, config, force_offline, judge)
        if args.verbose:
            text += "\n" + _format_details(rows)
        sections.append(text)

    settings = (
        f"Settings: compact_threshold_tokens={config.compact_threshold_tokens}, "
        f"compact_keep_messages={config.compact_keep_messages}, "
        f"profile_confidence_threshold={config.profile_confidence_threshold}\n"
    )
    report = "# Memory Benchmark\n\n" + settings + "\n" + "\n".join(sections)
    print(report)
    if args.output:
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
