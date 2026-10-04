from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_advanced import AdvancedAgent  # noqa: E402
from agent_baseline import BaselineAgent  # noqa: E402
from benchmark import load_conversations, recall_points, run_agent_benchmark  # noqa: E402
from config import load_config  # noqa: E402
from memory_store import (  # noqa: E402
    CompactMemoryManager,
    UserProfileStore,
    extract_profile_updates,
    validate_llm_fact,
)
from model_provider import normalize_provider  # noqa: E402

LONG_TURN = (
    "Mình kể thêm một đoạn dài về công việc MLOps: team hay giữ lại tất cả logs, tất cả context và "
    "tất cả prompt snippets vì sợ mất thông tin, nhưng kết quả là hệ thống vừa chậm vừa khó kiểm soát chi phí. "
)


def make_config(tmp_path: Path):
    """Isolated config: state lives in tmp_path, tiny compact threshold, never live."""

    base = load_config()
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return replace(
        base,
        state_dir=state_dir,
        compact_threshold_tokens=200,
        compact_keep_messages=2,
        live_mode=False,
    )


# -- User.md -------------------------------------------------------------------


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    # Missing file: read returns an empty default profile, size is 0, edit is a no-op.
    assert "## Profile" in store.read_text("dungct")
    assert store.file_size("dungct") == 0
    assert store.edit_text("dungct", "anything", "else") is False

    path = store.write_text("dungct", "# User.md: dungct\n\n## Profile\n- name: DũngCT\n")
    assert path.name == "User.md" and path.exists()
    assert store.facts("dungct") == {"name": "DũngCT"}
    size_before = store.file_size("dungct")
    assert size_before > 0

    assert store.edit_text("dungct", "- name: DũngCT", "- name: DũngCT\n- location: Huế") is True
    assert store.facts("dungct")["location"] == "Huế"
    assert store.edit_text("dungct", "text-that-is-not-there", "x") is False
    assert store.file_size("dungct") > size_before

    # User ids are sanitized so they cannot escape the profiles directory.
    assert store.path_for("../../etc").parent.parent == tmp_path / "profiles"


def test_advanced_agent_writes_user_markdown(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("dungct", "t1", "Chào bạn, mình tên là DũngCT.")
    agent.reply("dungct", "t1", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")

    content = agent.profile_store.read_text("dungct")
    assert agent.profile_store.path_for("dungct").exists()
    assert "- name: DũngCT" in content
    assert "- location: Đà Nẵng" in content
    assert "- profession: backend engineer" in content
    assert agent.memory_file_size("dungct") > 0


# -- compact memory --------------------------------------------------------------


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)

    agent.reply("u", "short", "Chào bạn.")
    assert agent.compaction_count("short") == 0  # short threads never compact

    for i in range(10):
        agent.reply("u", "long", f"Lượt {i}. {LONG_TURN}")
    context = agent.compact_memory.context("long")
    assert agent.compaction_count("long") >= 2
    assert len(context["messages"]) <= config.compact_keep_messages + 1
    assert "Lượt" in context["summary"]  # old turns survive as summary bullets
    assert agent.compact_memory.total_tokens("long") <= config.compact_threshold_tokens + 120


def test_compact_summary_stays_bounded() -> None:
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2, summary_max_items=3)
    for i in range(30):
        manager.append("t", "user", f"Tin số {i}: {LONG_TURN}")
    summary = str(manager.context("t")["summary"])
    assert len(summary.splitlines()) <= 3
    assert "Tin số 0:" not in summary  # oldest bullets roll off


# -- cross-session recall ------------------------------------------------------------


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    facts = ["Chào bạn, mình tên là DũngCT.", "Đồ uống yêu thích là cà phê sữa đá."]
    question = "Mình tên gì và đồ uống yêu thích là gì?"

    for turn in facts:
        baseline.reply("dungct", "session-1", turn)
        advanced.reply("dungct", "session-1", turn)

    # Baseline remembers inside the same thread...
    same_thread = baseline.reply("dungct", "session-1", question)["response"]
    assert recall_points(same_thread, ["DũngCT", "cà phê sữa đá"]) == 1.0
    # ...but forgets in a new thread.
    new_thread = baseline.reply("dungct", "session-2", question)["response"]
    assert "DũngCT" not in new_thread

    # Advanced recalls from User.md in a new thread, and even from a fresh agent instance.
    assert recall_points(advanced.reply("dungct", "session-2", question)["response"], ["DũngCT", "cà phê sữa đá"]) == 1.0
    restarted = AdvancedAgent(config, force_offline=True)
    assert "DũngCT" in restarted.reply("dungct", "session-3", question)["response"]


def test_correction_replaces_old_fact(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("dungct", "t1", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")
    agent.reply("dungct", "t2", "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.")
    agent.reply("dungct", "t2", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    answer = agent.reply("dungct", "t3", "Hiện tại mình làm nghề gì và đang ở đâu?")["response"]
    assert "Huế" in answer and "MLOps engineer" in answer
    assert "Đà Nẵng" not in answer and "backend" not in answer

    profile = agent.profile_store.read_text("dungct")
    profile_section = profile.split("## Corrections")[0]
    assert "Đà Nẵng" not in profile_section  # old value is not kept as a current fact
    assert "location: Đà Nẵng -> Huế" in profile


def test_questions_jokes_and_noise_are_not_saved() -> None:
    assert extract_profile_updates("Mình tên gì?") == {}
    assert extract_profile_updates("Nhắc lại giúp mình: tên, món ăn yêu thích và mình nuôi con gì.") == {}
    assert extract_profile_updates("Bạn có thể nhắc lại tên mình không?") == {}

    joke = (
        "Có lúc mình đùa với đồng nghiệp rằng hay là chuyển sang product manager cho đỡ phải ngồi canh pipeline, "
        "nhưng đó chỉ là câu đùa. Nghề nghiệp hiện tại vẫn là MLOps engineer."
    )
    assert extract_profile_updates(joke) == {"profession": "MLOps engineer"}

    noise = "Tương tự, Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày với đối tác chứ không phải nơi ở hiện tại."
    assert "location" not in extract_profile_updates(noise)

    hypothetical = "Nếu sau này mình có nhắc lại Đà Nẵng như ví dụ cũ thì đừng lấy nó làm nơi ở hiện tại nhé."
    assert "location" not in extract_profile_updates(hypothetical)


def test_confidence_threshold_blocks_low_confidence_writes(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.upsert_fact("u", "profession", "product manager", confidence=0.4, threshold=0.6) == "rejected"
    assert not store.exists("u")
    assert store.upsert_fact("u", "profession", "MLOps engineer", confidence=0.9, threshold=0.6) == "added"
    assert store.upsert_fact("u", "profession", "MLOps engineer", confidence=0.9, threshold=0.6) == "unchanged"
    assert store.upsert_fact("u", "unknown_field", "x", confidence=1.0) == "rejected"


def test_llm_tool_writes_are_guarded() -> None:
    """Regressions observed in the first live run (gpt-4o-mini via the save_user_fact tool)."""

    correction = "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa."
    # The LLM saved the negated old city, overwriting the correct "Huế".
    assert validate_llm_fact("location", "Đà Nẵng", correction) != ""
    assert validate_llm_fact("location", "Huế", correction, locked_fields={"location"}) != ""
    # The LLM stuffed whole sentences into multi-value fields.
    chatty = "Mình muốn mỗi conversation có ít nhất 10 lượt và nghe tự nhiên hơn."
    assert validate_llm_fact("response_style", "muốn mỗi conversation có ít nhất 10 lượt", chatty) != ""
    # Values must be stated by the user, not invented.
    assert validate_llm_fact("interests", "AI pipeline", "Mình thích Python.") != ""
    # Grounded but mis-filed overwrite ("pha cà phê" replacing "cà phê sữa đá"): only empty fields are fillable.
    morning = "Buổi sáng mình thường pha cà phê rồi đọc issue mới."
    assert validate_llm_fact("favorite_drink", "cà phê", morning, existing={"favorite_drink": "cà phê sữa đá"}) != ""
    # Multi-value fields are extractor-only (LLM additions evicted stable interests like "Python").
    assert validate_llm_fact("interests", "Python; RAG", "Mình đang học Python và RAG.") != ""
    # A short, verbatim, non-negated fact for an empty field passes.
    assert validate_llm_fact("pet", "corgi", "Mình nuôi một bé corgi tên Bơ.", existing={"name": "DũngCT"}) == ""


def test_corrections_log_is_capped(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles", max_superseded_history=3)
    for city in ["Huế", "Đà Nẵng", "Hà Nội", "Huế", "Cần Thơ", "Đà Lạt"]:
        store.upsert_fact("u", "location", city)
    corrections = store.read_text("u").split("## Corrections")[1].strip().splitlines()
    assert len(corrections) == 3
    assert store.facts("u")["location"] == "Đà Lạt"


# -- prompt load --------------------------------------------------------------------


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), compact_threshold_tokens=1000, compact_keep_messages=4)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    stress = load_conversations(config.data_dir / "advanced_long_context.json")[0]

    last_baseline = last_advanced = 0
    for turn in stress["turns"]:
        last_baseline = baseline.reply(stress["user_id"], "long", turn)["prompt_tokens"]
        last_advanced = advanced.reply(stress["user_id"], "long", turn)["prompt_tokens"]

    assert advanced.compaction_count("long") >= 2
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
    # Per-turn context is bounded for advanced, but keeps growing for baseline.
    assert last_advanced < last_baseline / 2


def test_short_threads_cost_more_for_advanced(tmp_path: Path) -> None:
    """Trade-off check: without compaction, injecting User.md is pure overhead."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for turn in ["Chào bạn, mình tên là DũngCT.", "Mình thích Python.", "Mình ở Huế."]:
        baseline.reply("u", "short", turn)
        advanced.reply("u", "short", turn)
    assert advanced.compaction_count("short") == 0
    assert advanced.prompt_token_usage("short") > baseline.prompt_token_usage("short")


# -- benchmark + providers -----------------------------------------------------------


def test_benchmark_separates_agents(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), compact_threshold_tokens=1000, compact_keep_messages=4)
    conversations = load_conversations(config.data_dir / "conversations.json")
    base = run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), conversations, config)
    adv = run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), conversations, config)

    assert base.recall_score == 0.0
    assert adv.recall_score >= 0.9
    assert base.memory_growth_bytes == 0 and adv.memory_growth_bytes > 0
    assert adv.response_quality > base.response_quality


@pytest.mark.parametrize(
    ("alias", "expected"),
    [("anthorpic", "anthropic"), ("Claude", "anthropic"), ("google", "gemini"), ("open_router", "openrouter"),
     ("openai-compatible", "custom"), ("ollama", "ollama"), ("openai", "openai")],
)
def test_normalize_provider(alias: str, expected: str) -> None:
    assert normalize_provider(alias) == expected


def test_normalize_provider_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        normalize_provider("not-a-provider")
