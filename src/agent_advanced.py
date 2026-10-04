from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_baseline import invoke_live
from config import LabConfig, load_config
from memory_store import (
    FIELD_ORDER,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_candidates,
    is_question,
    render_fact_answer,
    requested_fields,
    validate_llm_fact,
)
from model_provider import build_chat_model

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt có bộ nhớ dài hạn. Dùng User.md làm nguồn sự thật về người dùng; "
    "nếu có đính chính thì ưu tiên thông tin mới nhất. Không lưu câu hỏi, câu đùa hay giả định thành fact. "
    "Trả lời theo style trả lời đã lưu trong User.md."
)

LIVE_MEMORY_INSTRUCTIONS = (
    "User.md đã được hệ thống tự cập nhật từ tin nhắn mới nhất (kể cả đính chính). Chỉ gọi save_user_fact "
    "khi người dùng vừa nêu một fact ổn định cho field CÒN TRỐNG trong User.md, value trích nguyên văn ngắn."
)

# Short summary prompt: stable facts already live in User.md, so the summary only keeps the
# conversation gist. The default LangChain prompt asks for 4 sections and costs far more output.
LIVE_SUMMARY_PROMPT = (
    "Tóm tắt hội thoại dưới đây thành tối đa 6 gạch đầu dòng tiếng Việt (tổng dưới 120 từ). "
    "Giữ chủ đề chính, ý/pattern người dùng rút ra và các việc còn dang dở. "
    "Bỏ qua fact hồ sơ cá nhân (đã lưu ở User.md). Chỉ trả về các gạch đầu dòng.\n\n"
    "<messages>\n{messages}\n</messages>"
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: three memory layers.

    1. short-term: recent messages of the thread (kept verbatim)
    2. persistent: `User.md` per user (survives new threads / sessions)
    3. compact: older messages folded into a rolling summary when the thread is long
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(
            self.config.state_dir / "profiles", max_superseded_history=self.config.max_superseded_history
        )
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
            summary_max_items=self.config.summary_max_items,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        # Live mode: real SummarizationMiddleware runs, counted per thread.
        self.live_compactions: dict[str, int] = {}
        self._active_user_id: str | None = None
        # Evidence used to validate LLM tool writes during the current live turn.
        self._current_message = ""
        self._locked_fields: set[str] = set()
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        if self.langchain_agent is not None:
            return self.live_compactions.get(thread_id, 0)
        return self.compact_memory.compaction_count(thread_id)

    # -- shared memory steps -----------------------------------------------------

    def _persist_profile_updates(self, user_id: str, message: str) -> list[tuple[str, str, str]]:
        """Write guard-railed facts to User.md. Returns (field, value, status) for real changes."""

        changes = []
        for candidate in extract_profile_candidates(message):
            status = self.profile_store.upsert_fact(
                user_id,
                candidate.field,
                candidate.value,
                confidence=candidate.confidence,
                threshold=self.config.profile_confidence_threshold,
            )
            if status in ("added", "updated"):
                changes.append((candidate.field, candidate.value, status))
        return changes

    def _record_turn(self, thread_id: str, agent_tokens: int, prompt_tokens: int) -> None:
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

    # -- offline (deterministic) path -------------------------------------------

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1-2. extract stable facts and persist them to User.md (persistent layer)
        changes = self._persist_profile_updates(user_id, message)
        # 3. short-term layer; compaction happens inside append() when needed
        self.compact_memory.append(thread_id, "user", message)
        # 4. prompt load = system + User.md + summary + recent messages
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        # 5. answer from persisted memory
        response = self._offline_response(user_id, thread_id, message, changes)
        # 6. bookkeeping; memory writes are tool calls, so they cost agent tokens too
        self.compact_memory.append(thread_id, "assistant", response)
        memory_op_tokens = sum(estimate_tokens(f"upsert_fact({f}, {v})") for f, v, _ in changes)
        agent_tokens = estimate_tokens(message) + estimate_tokens(response) + memory_op_tokens
        self._record_turn(thread_id, agent_tokens, prompt_tokens)
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "profile_changes": changes,
            "compactions": self.compaction_count(thread_id),
            "mode": "offline",
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(ADVANCED_SYSTEM_PROMPT)
            + estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(str(context["summary"]))
            + sum(estimate_tokens(m["content"]) for m in context["messages"])
        )

    def _offline_response(
        self,
        user_id: str,
        thread_id: str,
        message: str,
        changes: list[tuple[str, str, str]] | None = None,
    ) -> str:
        facts = self.profile_store.facts(user_id)
        fields = requested_fields(message)
        if fields:
            return render_fact_answer(fields, facts)
        if is_question(message):
            # Open question about the user ("Bạn có biết DũngCT không?"): short profile card.
            known = [f for f in ("name", "profession", "location", "interests") if facts.get(f)]
            if known:
                return render_fact_answer(known, facts)
            return "Mình chưa có thông tin này trong bộ nhớ hiện tại."
        if changes:
            return "Đã cập nhật User.md: " + ", ".join(f"{f}={v}" for f, v, _ in changes) + "."
        return "Đã ghi nhận."

    # -- live (LangChain) path ----------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._active_user_id = user_id
        self._current_message = message
        # Fields the deterministic extractor handled this turn cannot be overwritten by the LLM tool.
        self._locked_fields = {
            c.field
            for c in extract_profile_candidates(message)
            if c.confidence >= self.config.profile_confidence_threshold
        }
        changes = self._persist_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)

        context = AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id)))
        response, usage = invoke_live(self.langchain_agent, thread_id, message, context=context)
        # Real usage covers tool round trips AND the summarization call of the middleware.
        prompt_tokens = usage.input_tokens or self._estimate_prompt_context_tokens(user_id, thread_id)
        self.live_compactions[thread_id] = self.live_compactions.get(thread_id, 0) + usage.summary_calls

        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = estimate_tokens(message) + (usage.output_tokens or estimate_tokens(response))
        self._record_turn(thread_id, agent_tokens, prompt_tokens)
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "profile_changes": changes,
            "compactions": self.compaction_count(thread_id),
            "mode": "live",
        }

    def _maybe_build_langchain_agent(self):
        """Live agent: tools for User.md + dynamic prompt + summarization middleware.

        Returns None (offline mode) unless `LAB_MODE=live` and credentials exist.
        """

        if not self.config.live_mode:
            return None
        try:
            from langchain.agents import create_agent
            from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
            from langchain.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None

        store = self.profile_store
        threshold = self.config.profile_confidence_threshold
        agent = self

        def current_user(runtime_context: Any = None) -> str:
            user_id = getattr(runtime_context, "user_id", None) or agent._active_user_id
            if not user_id:
                raise ValueError("No active user")
            return user_id

        # No read tool: User.md is already injected by the dynamic prompt, and every extra
        # tool schema / round trip is paid again in prompt tokens.
        @tool
        def save_user_fact(field: str, value: str, confidence: float) -> str:
            """Fill ONE missing stable fact about the user in User.md (field must currently be empty).

            field: name, location, profession, favorite_drink, favorite_food, pet.
            value: a short phrase copied verbatim from the user's latest message.
            Never save questions, jokes, hypotheticals or values the user says are no longer true.
            confidence: 0..1.
            """

            reason = validate_llm_fact(
                field, value, agent._current_message, agent._locked_fields, existing=store.facts(current_user())
            )
            if reason:
                return f"rejected: {reason}"
            status = store.upsert_fact(current_user(), field, value, confidence=confidence, threshold=threshold)
            return f"{status}: {field}={value}" if status != "rejected" else (
                f"rejected: confidence below {threshold} or unknown field ({', '.join(FIELD_ORDER)})"
            )

        @dynamic_prompt
        def inject_profile(request: ModelRequest) -> str:
            context = getattr(request.runtime, "context", None)
            profile = store.read_text(current_user(context))
            instructions = f"\n{LIVE_MEMORY_INSTRUCTIONS}" if agent.config.live_memory_tool else ""
            return f"{ADVANCED_SYSTEM_PROMPT}{instructions}\n\n<user_md>\n{profile}\n</user_md>"

        model = build_chat_model(self.config.model)
        threshold_tokens = self.config.compact_threshold_tokens
        return create_agent(
            model=model,
            tools=[save_user_fact] if self.config.live_memory_tool else [],
            middleware=[
                inject_profile,
                # Keep a token budget well below the trigger. With keep=("messages", 4) the kept
                # long turns alone were near the trigger, so the middleware re-summarized before
                # almost every model call (thrashing: 26 summaries in a 16-turn thread).
                SummarizationMiddleware(
                    model=model,
                    trigger=("tokens", threshold_tokens),
                    keep=("tokens", int(threshold_tokens * 0.3)),
                    summary_prompt=LIVE_SUMMARY_PROMPT,
                ),
            ],
            context_schema=AgentContext,
            checkpointer=InMemorySaver(),
        )
