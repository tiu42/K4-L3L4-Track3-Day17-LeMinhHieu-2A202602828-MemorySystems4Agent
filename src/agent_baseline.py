from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    MULTI_VALUE_FIELDS,
    estimate_tokens,
    extract_profile_updates,
    is_question,
    merge_multi_values,
    render_fact_answer,
    requested_fields,
)
from model_provider import UsageTracker, build_chat_model

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt. Trả lời ngắn gọn dựa trên hội thoại hiện tại. "
    "Bạn không có bộ nhớ dài hạn giữa các phiên."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


def message_text(message: Any) -> str:
    """Plain text of a LangChain message whose content may be a list of blocks."""

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def invoke_live(agent: Any, thread_id: str, message: str, context: Any = None) -> tuple[str, UsageTracker]:
    """Run one live turn and return (reply text, real token usage of every LLM call in it)."""

    tracker = UsageTracker()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}, "callbacks": [tracker]},
        context=context,
    )
    return message_text(result["messages"][-1]), tracker


class BaselineAgent:
    """Agent A: within-session memory only.

    - Keeps the full message list per `thread_id` (no compaction, no `User.md`).
    - A new thread starts with an empty session, so long-term facts are forgotten.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    @property
    def mode(self) -> str:
        return "live" if self.langchain_agent is not None else "offline"

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the response plus token accounting for this turn.

        `user_id` is accepted for API parity with the advanced agent but never
        used for memory: the baseline must not remember across threads.
        """

        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline has no persistent memory file.
        return 0

    # -- offline (deterministic) path ------------------------------------------

    def _prompt_tokens(self, session: SessionState) -> int:
        # The baseline drags the *entire* thread history into every prompt.
        return estimate_tokens(BASELINE_SYSTEM_PROMPT) + sum(estimate_tokens(m["content"]) for m in session.messages)

    def _session_facts(self, session: SessionState) -> dict[str, str]:
        """Facts visible *inside this thread only* (re-read from history, never persisted)."""

        facts: dict[str, str] = {}
        threshold = self.config.profile_confidence_threshold
        for msg in session.messages:
            if msg["role"] != "user":
                continue
            for name, value in extract_profile_updates(msg["content"], threshold).items():
                if name in MULTI_VALUE_FIELDS and name in facts:
                    facts[name] = merge_multi_values(facts[name], value)
                else:
                    facts[name] = value
        return facts

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = self._prompt_tokens(session)

        fields = requested_fields(message)
        if fields:
            response = render_fact_answer(fields, self._session_facts(session))
        elif is_question(message):
            response = "Mình chỉ dựa được vào nội dung trong phiên chat hiện tại."
        else:
            response = "Đã ghi nhận trong phiên này."

        session.messages.append({"role": "assistant", "content": response})
        turn_tokens = estimate_tokens(message) + estimate_tokens(response)
        session.token_usage += turn_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {
            "response": response,
            "agent_tokens": turn_tokens,
            "prompt_tokens": prompt_tokens,
            "mode": "offline",
        }

    # -- live (LangChain) path -------------------------------------------------

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        response, usage = invoke_live(self.langchain_agent, thread_id, message)
        # Real provider usage when reported; heuristic estimate otherwise.
        prompt_tokens = usage.input_tokens or self._prompt_tokens(session)

        session.messages.append({"role": "assistant", "content": response})
        turn_tokens = estimate_tokens(message) + (usage.output_tokens or estimate_tokens(response))
        session.token_usage += turn_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"response": response, "agent_tokens": turn_tokens, "prompt_tokens": prompt_tokens, "mode": "live"}

    def _maybe_build_langchain_agent(self):
        """Live agent = `create_agent` + `InMemorySaver` (thread-scoped memory only).

        Returns None (offline mode) unless `LAB_MODE=live` and credentials exist.
        """

        if not self.config.live_mode:
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None

        return create_agent(
            model=build_chat_model(self.config.model),
            tools=[],
            system_prompt=BASELINE_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )
