from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator (~4 characters per token).

    Not tokenizer-accurate, but stable and deterministic, which is what the
    offline benchmark needs to compare agents fairly.
    """

    text = (text or "").strip()
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def normalize_text(text: str) -> str:
    """NFC-normalize so composed / decomposed Vietnamese compare equal."""

    return unicodedata.normalize("NFC", text or "")


# ---------------------------------------------------------------------------
# Structured profile schema
# ---------------------------------------------------------------------------

# Single-value fields: a newer value replaces the old one (conflict handling).
SINGLE_VALUE_FIELDS = ("name", "location", "profession", "favorite_drink", "favorite_food", "pet")
# Multi-value fields: values are merged as a de-duplicated, capped list.
MULTI_VALUE_FIELDS = ("response_style", "interests")
FIELD_ORDER = SINGLE_VALUE_FIELDS + MULTI_VALUE_FIELDS
MULTI_VALUE_SEPARATOR = "; "
MAX_MULTI_VALUES = 6

FIELD_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "response_style": "Style trả lời",
    "interests": "Mối quan tâm kỹ thuật",
}


@dataclass
class FactCandidate:
    """One extracted fact plus how sure the extractor is about it."""

    field: str
    value: str
    confidence: float
    evidence: str = ""


# ---------------------------------------------------------------------------
# Persistent memory: User.md
# ---------------------------------------------------------------------------

_FACT_LINE = re.compile(r"^- (?P<field>[a-z_]+): (?P<value>.+?)(?: \(confidence: (?P<conf>[0-9.]+)\))?\s*$")


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` (one markdown file per user).

    Layout of the file:

        # User.md: <user_id>
        ## Profile        <- current, de-conflicted facts
        ## Corrections    <- bounded log of superseded values
    """

    root_dir: Path
    max_superseded_history: int = 5

    # -- raw file operations -------------------------------------------------

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", normalize_text(user_id).strip()).strip("_") or "anonymous"
        return Path(self.root_dir) / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return self._render(user_id, {}, [])

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        if not path.exists() or not search_text:
            return False
        content = path.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def exists(self, user_id: str) -> bool:
        return self.path_for(user_id).exists()

    def delete(self, user_id: str) -> None:
        path = self.path_for(user_id)
        if path.exists():
            path.unlink()

    # -- structured helpers --------------------------------------------------

    def facts(self, user_id: str) -> dict[str, str]:
        return {name: value for name, (value, _) in self._parse(self.read_text(user_id))[0].items()}

    def upsert_fact(
        self,
        user_id: str,
        field_name: str,
        value: str,
        confidence: float = 1.0,
        threshold: float = 0.0,
    ) -> str:
        """Insert or update one fact. Returns added / updated / unchanged / rejected.

        Guardrails:
        - confidence threshold: low-confidence facts never reach disk
        - conflict handling: single-value fields keep only the newest value;
          the old one moves to a bounded `Corrections` log
        - multi-value fields are merged and capped to limit file growth
        """

        value = normalize_text(value).strip()
        if field_name not in FIELD_ORDER or not value or confidence < threshold:
            return "rejected"

        facts, corrections = self._parse(self.read_text(user_id))
        old = facts.get(field_name)

        if field_name in MULTI_VALUE_FIELDS:
            merged = merge_multi_values(old[0] if old else "", value)
            if old and merged == old[0]:
                return "unchanged"
            facts[field_name] = (merged, max(confidence, old[1] if old else 0.0))
        else:
            if old and old[0].casefold() == value.casefold():
                return "unchanged"
            facts[field_name] = (value, confidence)
            if old:
                corrections.append(f"{field_name}: {old[0]} -> {value}")
                corrections = corrections[-self.max_superseded_history :]

        self.write_text(user_id, self._render(user_id, facts, corrections))
        return "updated" if old else "added"

    # -- (de)serialization ---------------------------------------------------

    @staticmethod
    def _parse(content: str) -> tuple[dict[str, tuple[str, float]], list[str]]:
        facts: dict[str, tuple[str, float]] = {}
        corrections: list[str] = []
        section = ""
        for line in content.splitlines():
            if line.startswith("## "):
                section = line[3:].strip().lower()
                continue
            if not line.startswith("- "):
                continue
            if section == "profile":
                match = _FACT_LINE.match(line)
                if match:
                    facts[match["field"]] = (match["value"].strip(), float(match["conf"] or 1.0))
            elif section == "corrections":
                corrections.append(line[2:].strip())
        return facts, corrections

    @staticmethod
    def _render(user_id: str, facts: dict[str, tuple[str, float]], corrections: list[str]) -> str:
        lines = [
            f"# User.md: {user_id}",
            "",
            "> Persistent memory. Chỉ lưu fact ổn định, đủ confidence; fact mới ghi đè fact cũ.",
            "",
            "## Profile",
        ]
        ordered = [f for f in FIELD_ORDER if f in facts] + sorted(f for f in facts if f not in FIELD_ORDER)
        for name in ordered:
            value, confidence = facts[name]
            lines.append(f"- {name}: {value} (confidence: {confidence:.2f})")
        if corrections:
            lines += ["", "## Corrections"]
            lines += [f"- {item}" for item in corrections]
        return "\n".join(lines) + "\n"


def merge_multi_values(existing: str, incoming: str) -> str:
    items = [x.strip() for x in existing.split(MULTI_VALUE_SEPARATOR) if x.strip()]
    for item in (x.strip() for x in incoming.split(MULTI_VALUE_SEPARATOR)):
        if not item or any(item.casefold() == x.casefold() for x in items):
            continue
        # "3 bullet" is a more specific version of the generic "dạng bullet".
        if "bullet" in item:
            if item[0].isdigit():
                items = [x for x in items if "bullet" not in x]
            elif any("bullet" in x for x in items):
                continue
        items.append(item)
    return MULTI_VALUE_SEPARATOR.join(items[-MAX_MULTI_VALUES:])


# ---------------------------------------------------------------------------
# Fact extraction (rule-based entity extraction + confidence scoring)
# ---------------------------------------------------------------------------

_CITIES = (
    "Đà Nẵng", "Huế", "Hà Nội", "Hồ Chí Minh", "TP.HCM", "Sài Gòn", "Hải Phòng",
    "Cần Thơ", "Nha Trang", "Đà Lạt", "Quy Nhơn", "Vũng Tàu",
)
_CITY_ALT = "|".join(re.escape(c) for c in sorted(_CITIES, key=len, reverse=True))

_LOCATION_PATTERNS = [
    # "mình ở / mình đang ở / mình vẫn ở / mình đang làm việc ở"
    re.compile(
        rf"\b(?:mình|tôi|em)\s+(?:(?:vẫn|đang|hiện|giờ|hiện\s+tại|đã)\s+)*"
        rf"(?:ở|sống\s+(?:ở|tại)|làm\s+việc\s+(?:ở|tại))\s+(?P<v>{_CITY_ALT})",
        re.I,
    ),
    re.compile(rf"\bhiện\s+(?:đang\s+)?ở\s+(?P<v>{_CITY_ALT})", re.I),
    re.compile(rf"nơi\s+ở(?:\s+hiện\s+tại)?\s+(?:là|:)\s*(?P<v>{_CITY_ALT})", re.I),
    re.compile(rf"nơi\s+ở[^.]*?\btừ\s+(?:{_CITY_ALT})\s+sang\s+(?P<v>{_CITY_ALT})", re.I),
]

_ROLE = r"[A-Za-z][\w-]*\s+(?:engineer|developer|manager|scientist|designer|analyst|researcher|architect)"
_PROFESSION_PATTERN = re.compile(rf"(?:\blàm|\blà|\bsang|\bnghề(?:\s+nghiệp)?)\s+(?P<v>{_ROLE})\b", re.I)

_NAME_PREFIXES = [
    re.compile(r"\b(?:mình|tôi|em)\s+tên(?:\s+là)?\s+", re.I),
    re.compile(r"\btên\s+(?:của\s+)?(?:mình|tôi)\s+là\s+", re.I),
    re.compile(r"(?:^|[:;,]\s*)tên\s+(?=[^\W\d_])", re.I),
]

_FAVORITE_PATTERNS = {
    "favorite_drink": re.compile(r"đồ\s+uống\s+(?:yêu|ưa)\s+thích(?:\s+của\s+mình)?\s+(?:là|:)\s+(?P<v>[^.,;!?]+)", re.I),
    "favorite_food": re.compile(r"món(?:\s+ăn)?\s+(?:yêu|ưa)\s+thích(?:\s+của\s+mình)?\s+(?:là|:)\s+(?P<v>[^.,;!?]+)", re.I),
}
_PET_PATTERN = re.compile(r"\bnuôi\s+(?:một\s+)?(?:bé\s+|con\s+)?(?P<kind>[^\W\d_]+)(?:\s+tên\s+(?P<name>[^\W\d_]+))?", re.I)

_TECH_INTERESTS = ("Python", "AI ứng dụng", "AI agent", "MLOps", "RAG", "evaluation", "memory architecture")
_INTEREST_TRIGGER = re.compile(r"\b(?:thích|quan tâm|học|ôn|đọc về|đam mê)\b", re.I)

_STYLE_TRIGGER = re.compile(r"trả\s+lời|style|giải\s+thích|trình\s+bày", re.I)
_STYLE_TAGS = [
    (re.compile(r"ngắn|gọn|lan\s+man", re.I), "ngắn gọn"),
    (re.compile(r"cấu\s+trúc", re.I), "có cấu trúc"),
    (re.compile(r"ví\s+dụ\s+thực\s+(?:tế|chiến)", re.I), "có ví dụ thực tế"),
    (re.compile(r"trade-off", re.I), "nhấn trade-off"),
    (re.compile(r"định\s+lượng|số\s+liệu", re.I), "có số liệu minh họa"),
]
_BULLET_COUNT = re.compile(r"(\d+)\s+bullet", re.I)

# Hedges / hypotheticals / jokes (a sentence *starting* with "Nếu" is conditional): the sentence may mention a value that is NOT a fact.
_HEDGE = re.compile(r"^\s*nếu\b|\bhay\s+là\b|\bđùa\b|\bcó\s+lẽ\b|\bgiả\s+sử\b|\bví\s+dụ\s+cũ\b", re.I)
_JOKE = re.compile(r"\bđùa\b|\bgiả\s+sử\b", re.I)
# Explicit corrections / confirmations raise confidence slightly.
_REINFORCE = re.compile(r"đính\s+chính|cập\s+nhật|hiện\s+tại|thực\s+ra|\bgiờ\b|nhớ\s+là|nhắc\s+lại", re.I)
# A value directly preceded by a negation is an *old* / rejected value.
_NEGATION_BEFORE = re.compile(r"(?:không\s+còn|không\s+phải|chứ\s+không|đừng|không)\s+(?:\S+\s+)?$", re.I)

_QUESTION = re.compile(
    r"\?\s*$|\bgì\b|\bở\s+đâu\b|\bthế\s+nào\b|\bbao\s+nhiêu\b|\blà\s+ai\b"
    r"|^\s*bạn\s+có\s+biết\b",
    re.I,
)

# Interrogative words are never valid fact values (case-sensitive: "AI" is not "ai").
_INTERROGATIVE_VALUE = re.compile(r"(?<!\w)(?:gì|nào|đâu|ai)(?!\w)")

_BASE_CONFIDENCE = {
    "name": 0.95,
    "location": 0.85,
    "profession": 0.85,
    "favorite_drink": 0.9,
    "favorite_food": 0.9,
    "pet": 0.85,
    "response_style": 0.8,
    "interests": 0.75,
}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", normalize_text(text).strip())
    return [p.strip() for p in parts if p.strip()]


def is_question(text: str) -> bool:
    return bool(_QUESTION.search(normalize_text(text).strip()))


def _is_negated(sentence: str, start: int) -> bool:
    return bool(_NEGATION_BEFORE.search(sentence[max(0, start - 20) : start]))


def _clean_value(value: str) -> str:
    value = re.sub(r"\s+(?:nhé|nha|ạ|đó|như cũ)$", "", value.strip(), flags=re.I)
    return value.strip(" .,:;!-")


def _extract_name(sentence: str) -> str | None:
    for prefix in _NAME_PREFIXES:
        for match in prefix.finditer(sentence):
            words: list[str] = []
            for raw in sentence[match.end() :].split():
                word = raw.strip(".,;:!?()\"'")
                if not word or not word[0].isupper():
                    break
                words.append(word)
                if raw != word:  # trailing punctuation ends the name
                    break
            if words:
                return " ".join(words)
    return None


def _sentence_candidates(sentence: str) -> list[FactCandidate]:
    found: list[FactCandidate] = []

    name = _extract_name(sentence)
    if name:
        found.append(FactCandidate("name", name, _BASE_CONFIDENCE["name"], sentence))

    locations = []
    for pattern in _LOCATION_PATTERNS:
        for match in pattern.finditer(sentence):
            if not _is_negated(sentence, match.start()):
                locations.append((match.start("v"), match["v"]))
    if locations:
        _, city = max(locations)  # the last-mentioned city in a sentence is the newest
        canonical = next(c for c in _CITIES if c.casefold() == city.casefold())
        found.append(FactCandidate("location", canonical, _BASE_CONFIDENCE["location"], sentence))

    roles = [
        (m.start("v"), m["v"])
        for m in _PROFESSION_PATTERN.finditer(sentence)
        if not _is_negated(sentence, m.start())
    ]
    if roles:
        found.append(FactCandidate("profession", max(roles)[1], _BASE_CONFIDENCE["profession"], sentence))

    for field_name, pattern in _FAVORITE_PATTERNS.items():
        match = pattern.search(sentence)
        if match:
            value = _clean_value(match["v"])
            if value and not re.search(r"\bgì\b", value, re.I):
                found.append(FactCandidate(field_name, value, _BASE_CONFIDENCE[field_name], sentence))

    pet = _PET_PATTERN.search(sentence)
    if pet:
        value = pet["kind"] + (f" tên {pet['name']}" if pet["name"] else "")
        found.append(FactCandidate("pet", value, _BASE_CONFIDENCE["pet"], sentence))

    if _STYLE_TRIGGER.search(sentence):
        tags = [tag for pattern, tag in _STYLE_TAGS if pattern.search(sentence)]
        bullet = _BULLET_COUNT.search(sentence)
        if bullet:
            tags.append(f"{bullet.group(1)} bullet")
        elif re.search(r"bullet", sentence, re.I):
            tags.append("dạng bullet")
        if tags:
            value = MULTI_VALUE_SEPARATOR.join(tags)
            found.append(FactCandidate("response_style", value, _BASE_CONFIDENCE["response_style"], sentence))

    if _INTEREST_TRIGGER.search(sentence):
        interests = [t for t in _TECH_INTERESTS if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", sentence)]
        if interests:
            value = MULTI_VALUE_SEPARATOR.join(interests)
            found.append(FactCandidate("interests", value, _BASE_CONFIDENCE["interests"], sentence))

    found = [c for c in found if not _INTERROGATIVE_VALUE.search(c.value)]

    # Confidence adjustments (bonus: confidence threshold before writing User.md).
    hedged = bool(_HEDGE.search(sentence))
    joking = bool(_JOKE.search(sentence))
    reinforced = bool(_REINFORCE.search(sentence))
    for candidate in found:
        preference = candidate.field in MULTI_VALUE_FIELDS
        if joking or (hedged and not preference):
            candidate.confidence -= 0.45
        if reinforced:
            candidate.confidence += 0.05
        candidate.confidence = round(min(0.99, max(0.0, candidate.confidence)), 2)
    return found


def extract_profile_candidates(message: str) -> list[FactCandidate]:
    """Extract every candidate fact (with confidence) from a user message.

    Question and recall-request sentences are skipped entirely, so neither
    "Mình tên gì?" nor "Nhắc lại giúp mình ... mình nuôi con gì." becomes a fact.
    """

    candidates: list[FactCandidate] = []
    for sentence in split_sentences(message):
        if is_question(sentence) or _RECALL_REQUEST.search(sentence):
            continue
        candidates.extend(_sentence_candidates(sentence))
    return candidates


MAX_LLM_ITEM_WORDS = 6


def validate_llm_fact(
    field_name: str,
    value: str,
    evidence: str,
    locked_fields: set[str] | frozenset = frozenset(),
    existing: dict[str, str] | None = None,
) -> str:
    """Guardrail for facts the *LLM* wants to save via a tool. Returns "" if OK, else the reason.

    Live benchmark showed LLM-written memory overwriting correct facts ("Huế" -> "Đà Nẵng"
    from "không còn ở Đà Nẵng") and stuffing whole sentences into multi-value fields.
    - trust levels: the LLM may only fill an EMPTY single-value field. Overwrites (corrections)
      and multi-value fields stay with the negation/joke-aware extractor, because grounded but
      mis-filed LLM writes ("favorite_food: đi biển Mỹ Khê chụp ảnh") still happened.
    - locked: the deterministic extractor already handled this field in this turn
    - not negated: "chứ không còn ở Đà Nẵng" does not ground "Đà Nẵng"
    - grounded: every item must literally appear in the user's current message
    - short: each item has at most MAX_LLM_ITEM_WORDS words
    """

    if field_name not in FIELD_ORDER:
        return f"unknown field (allowed: {', '.join(FIELD_ORDER)})"
    if field_name in MULTI_VALUE_FIELDS:
        return "multi-value fields are managed by the extractor"
    if (existing or {}).get(field_name):
        return "field already set; corrections are handled by the extractor"
    if field_name in locked_fields:
        return "field already updated from this message"
    source = normalize_text(evidence).casefold()
    for item in (x.strip() for x in normalize_text(value).split(";")):
        if not item:
            continue
        if len(item.split()) > MAX_LLM_ITEM_WORDS:
            return f"value too long: '{item}' (max {MAX_LLM_ITEM_WORDS} words)"
        positions = [m.start() for m in re.finditer(re.escape(item.casefold()), source)]
        if not positions:
            return f"value not stated by the user: '{item}'"
        if all(_is_negated(source, pos) for pos in positions):
            return f"value is negated by the user: '{item}'"
        if _INTERROGATIVE_VALUE.search(item):
            return f"value looks like a question word: '{item}'"
    return ""


def extract_profile_updates(message: str, threshold: float = 0.6) -> dict[str, str]:
    """Convert raw user text into stable profile facts above `threshold`.

    Later sentences win for single-value fields (corrections); multi-value
    fields are merged.
    """

    updates: dict[str, str] = {}
    for candidate in extract_profile_candidates(message):
        if candidate.confidence < threshold:
            continue
        if candidate.field in MULTI_VALUE_FIELDS and candidate.field in updates:
            updates[candidate.field] = merge_multi_values(updates[candidate.field], candidate.value)
        else:
            updates[candidate.field] = candidate.value
    return updates


# ---------------------------------------------------------------------------
# Answering recall questions from facts (shared by both offline agents)
# ---------------------------------------------------------------------------

_FIELD_QUERIES = [
    ("name", re.compile(r"\btên\b|\blà\s+ai\b", re.I)),
    ("location", re.compile(r"ở\s+đâu|nơi\s+ở|còn\s+ở|đang\s+ở", re.I)),
    ("profession", re.compile(r"\bnghề\b", re.I)),
    ("favorite_drink", re.compile(r"đồ\s+uống", re.I)),
    ("favorite_food", re.compile(r"món\s+ăn", re.I)),
    ("pet", re.compile(r"\bnuôi\b", re.I)),
    ("response_style", re.compile(r"style|kiểu\s+trả\s+lời|cách\s+trả\s+lời", re.I)),
    ("interests", re.compile(r"quan\s+tâm|sở\s+thích\s+kỹ\s+thuật", re.I)),
]


_RECALL_REQUEST = re.compile(
    r"^\s*nhắc\s+lại\s+(?!lần)|nhắc\s+lại\s+(?:giúp|cho\s+mình|xem)|\bhãy\s+nhắc\b|nhớ\s+lại|^\s*tóm\s+tắt", re.I
)


def requested_fields(message: str) -> list[str]:
    """Fields the user asks the agent to recall (only from question / recall-request sentences)."""

    asked: list[str] = []
    for sentence in split_sentences(message):
        if not (is_question(sentence) or _RECALL_REQUEST.search(sentence)):
            continue
        asked += [name for name, pattern in _FIELD_QUERIES if pattern.search(sentence) and name not in asked]
    return [name for name, _ in _FIELD_QUERIES if name in asked]


def render_fact_answer(fields: list[str], facts: dict[str, str]) -> str:
    """Answer only from stored facts; never echo the question (keeps recall honest)."""

    known = [(f, facts[f]) for f in fields if facts.get(f)]
    if not known:
        return "Mình chưa có thông tin này trong bộ nhớ hiện tại."
    lines = [f"{FIELD_LABELS[f]}: {v}" for f, v in known]
    missing = [FIELD_LABELS[f] for f in fields if not facts.get(f)]
    if missing:
        lines.append(f"Chưa có thông tin: {', '.join(missing)}")
    if "bullet" in facts.get("response_style", ""):
        return "\n".join(f"- {line}" for line in lines)
    return "; ".join(lines) + "."


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------


def _snippet(text: str, max_words: int = 18) -> str:
    sentences = split_sentences(text)
    first = sentences[0] if sentences else normalize_text(text).strip()
    words = first.split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary of older messages: one short bullet per user turn.

    Assistant turns in offline mode are acknowledgements, so they are dropped.
    Only the latest `max_items` bullets are kept, which bounds summary size.
    Existing summary bullets can be passed in as role="summary" messages.
    """

    bullets: list[str] = []
    for message in messages:
        role, content = message.get("role", ""), message.get("content", "")
        if role == "summary":
            bullets.extend(line for line in content.splitlines() if line.startswith("- "))
        elif role == "user" and content.strip():
            bullets.append(f"- user: {_snippet(content)}")
    return "\n".join(bullets[-max_items:])


@dataclass
class CompactMemoryManager:
    """Short-term memory with compaction for long threads.

    - Recent `keep_messages` messages stay verbatim.
    - When messages + summary exceed `threshold_tokens`, older messages are
      folded into a rolling summary and `compactions` is incremented.
    """

    threshold_tokens: int
    keep_messages: int
    summary_max_items: int = 8
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        thread["messages"].append({"role": role, "content": content})
        if self.total_tokens(thread_id) > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            self._compact(thread)

    def _compact(self, thread: dict[str, object]) -> None:
        messages: list[dict[str, str]] = thread["messages"]
        older, recent = messages[: -self.keep_messages], messages[-self.keep_messages :]
        previous = [{"role": "summary", "content": thread["summary"]}] if thread["summary"] else []
        thread["summary"] = summarize_messages(previous + older, max_items=self.summary_max_items)
        thread["messages"] = recent
        thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self._thread(thread_id)

    def total_tokens(self, thread_id: str) -> int:
        thread = self._thread(thread_id)
        return estimate_tokens(thread["summary"]) + sum(estimate_tokens(m["content"]) for m in thread["messages"])

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
