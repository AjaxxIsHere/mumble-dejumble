"""Noise injectors: each turns a clean text into a messy spoken version.

Contract (pinned by tests/test_injectors.py):
- Every injector is deterministic for a fixed rng and guarantees a change.
- Injectors never mutate an entity's internal characters; fillers and
  repetitions never split entity spans (values stay extractable).
- Corrections keep the invariant: abandoned value present in input, kept
  value == the clean text's verbatim span, abandoned != kept.
- "I was going to say X, but Y" is deliberately NOT a correction marker: that
  surface form belongs to the preservation category (spec §7 keeps it), so
  using it for both would teach contradictory targets.
"""

import random
import re
from dataclasses import dataclass, field

from data.generators.entities import (
    PoolSampler,
    SlotError,
    entity_spans,
    entity_variant,
    make_date,
    make_name,
    make_time,
    pick_entity_span,
)

FILLERS = (
    "um", "uh", "er", "hmm", "like", "you know", "I mean", "sort of", "kind of",
    "well", "basically", "right", "so", "okay so", "ah", "err", "I think",
    "you know what", "I guess",
)

_CORRECTION_MARKERS = {
    "easy": ("no wait", "actually", "I mean", "sorry", "wait"),
    "medium": ("no wait", "actually", "I mean", "sorry", "wait", "make that",
               "change that to", "rather", "I meant", "not"),
    "hard": ("no wait", "actually", "I mean", "sorry", "wait", "make that",
             "change that to", "rather", "I meant", "not", "oh hold on"),
}

_REFORMULATION_RESTART_MARKERS = (
    "I mean", "wait", "sorry", "actually", "let me rephrase that", "what I meant is",
)

_HEDGES = (
    "I wanted to say that maybe we could", "I was thinking maybe we should",
    "what I meant to say was", "so the thing is", "you know what", "hmm",
)
_DISTRACTORS = (
    "discuss the budget first", "go over last week's notes", "review the slides again",
    "talk about the old plan", "check the previous version", "walk through the details again",
    "look at the earlier draft", "compare the two options", "revisit the original idea",
    "run through the numbers once more", "go back to the start", "double-check the dates",
)
_DISTRACT_MARKERS = ("but actually", "anyway", "so actually", "I mean")

# Robust terminator stripping: questions/exclamations must not collide with the
# distractor marker ("...?, actually, ..." is invalid punctuation).
_TERMINATOR_RE = re.compile(r"[.!?]+$")


def _strip_terminator(text: str) -> str:
    return _TERMINATOR_RE.sub("", text).strip()


@dataclass
class InjectorResult:
    text: str
    delta: dict


def _tokens(text: str) -> list[tuple[str, int, int]]:
    """Whitespace tokens with (token, start, end) offsets."""
    out = []
    for m in re.finditer(r"\S+", text):
        out.append((m.group(0), m.start(), m.end()))
    return out


def _lower_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def _title_item(item: str) -> str:
    """Sentence-capitalize an item's first letter without mangling acronyms
    ("HDMI cable" stays "HDMI cable"; "milk" becomes "Milk")."""
    return item[:1].upper() + item[1:]


# --------------------------------------------------------------------------
# Filler
# --------------------------------------------------------------------------

def apply_filler(
    text: str, rng: random.Random, difficulty: str = "easy",
    protected: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
) -> InjectorResult:
    """Insert 1-4 fillers at token boundaries, never inside entity spans."""
    count = {"easy": 1, "medium": rng.randint(1, 2), "hard": rng.randint(2, 4)}[difficulty]
    tokens = _tokens(text)
    entities = entity_spans(text)
    protected = list(protected)

    gaps: list[int] = []
    for i in range(len(tokens) - 1):
        tok, start, end = tokens[i]
        nxt, nstart, _ = tokens[i + 1]
        if not nxt[0].isalnum():
            continue  # never insert before punctuation
        if any(end >= ps and nstart <= pe for ps, pe in protected):
            if difficulty != "hard":
                continue  # at hard, filler inside a correction cluster is allowed
        if any(end >= s.start and nstart <= s.end for s in entities):
            continue  # never split an entity span
        gaps.append(end)

    fillers = rng.sample(FILLERS, min(count, len(FILLERS)))
    insertions: list[tuple[int, str]] = []
    if gaps:
        # Never insert more fillers than there are gaps (short texts).
        for filler in fillers[: min(count, len(gaps))]:
            off = rng.choice(gaps)
            gaps.remove(off)
            insertions.append((off, f", {filler},"))
        for off, ins in sorted(insertions, key=lambda x: -x[0]):
            text = text[:off] + ins + text[off:]
    else:
        # fallback: sentence start ("Um, send the files ...")
        first = fillers[0]
        insertions.append((0, f"{first.capitalize()}, "))
        text = f"{first.capitalize()}, {_lower_first(text)}"

    return InjectorResult(text, {"transformation": "filler", "filler_count": len(insertions)})


# --------------------------------------------------------------------------
# Repetition
# --------------------------------------------------------------------------

def apply_repetition(
    text: str, rng: random.Random, difficulty: str = "easy",
    protected: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
) -> InjectorResult:
    """Word / phrase / partial-word stutter / clause echo.

    Targets are chosen on the ORIGINAL text (avoiding protected + entity
    spans), then applied right-to-left so offsets stay valid.
    """
    count = {"easy": 1, "medium": rng.randint(1, 2), "hard": rng.randint(2, 3)}[difficulty]
    tokens = _tokens(text)
    entities = entity_spans(text)
    protected = list(protected)

    def is_blocked(start: int, end: int) -> bool:
        if any(start < pe and end > ps for ps, pe in protected):
            return True
        if any(start < s.end and end > s.start for s in entities):
            return True
        return False

    replacements: list[tuple[int, int, str]] = []
    repeated: list[str] = []
    used_tokens: set[int] = set()

    for _ in range(count):
        replacement = _pick_repetition(text, tokens, rng, used_tokens, is_blocked)
        if replacement is None:
            break
        start, end, new, label = replacement
        replacements.append((start, end, new))
        repeated.append(label)

    if not replacements:
        # guaranteed-change fallback: stutter the first token
        tok, start, end = tokens[0]
        replacements = [(start, end, f"{tok} {tok}")]
        repeated = [tok]

    new_text = text
    for start, end, new in sorted(replacements, key=lambda x: -x[0]):
        new_text = new_text[:start] + new + new_text[end:]

    return InjectorResult(new_text, {"transformation": "repetition", "repeated_spans": repeated})


def _pick_repetition(text, tokens, rng, used_tokens, is_blocked):
    """Pick one repetition target on the original text; None if no candidate."""
    word_tokens = [
        i for i, (tok, s, e) in enumerate(tokens)
        if tok[0].isalnum() and i not in used_tokens and not is_blocked(s, e)
    ]
    if not word_tokens:
        return None

    flavor = rng.random()
    if flavor < 0.40 or len(word_tokens) < 2:  # word stutter
        i = rng.choice(word_tokens)
        tok, start, end = tokens[i]
        used_tokens.add(i)
        return start, end, f"{tok} {tok}", tok
    if flavor < 0.65 and len(tokens) >= 4:  # phrase repetition
        phrase_candidates = []
        for k in (3, 2):
            for i in range(len(tokens) - k + 1):
                seg = tokens[i:i + k]
                if all(t[0].isalnum() for t in seg) and all(
                    i + j not in used_tokens and not is_blocked(t[1], t[2])
                    for j, t in enumerate(seg)
                ):
                    phrase_candidates.append((i, k))
        if phrase_candidates:
            i, k = rng.choice(phrase_candidates)
            seg = tokens[i:i + k]
            phrase = " ".join(t[0] for t in seg)
            for j in range(i, i + k):
                used_tokens.add(j)
            return seg[0][1], seg[-1][2], f"{phrase} {phrase}", phrase
        i = rng.choice(word_tokens)
        tok, start, end = tokens[i]
        used_tokens.add(i)
        return start, end, f"{tok} {tok}", tok
    if flavor < 0.85:  # partial-word stutter
        partial_candidates = [
            i for i in word_tokens if len(tokens[i][0]) >= 4 and tokens[i][0].isalpha()
        ]
        if partial_candidates:
            i = rng.choice(partial_candidates)
            tok, start, end = tokens[i]
            used_tokens.add(i)
            prefix = tok[: rng.randint(1, 2)]
            return start, end, f"{prefix}-{tok}", f"{prefix}-{tok}"
        i = rng.choice(word_tokens)
        tok, start, end = tokens[i]
        used_tokens.add(i)
        return start, end, f"{tok} {tok}", tok
    # clause echo (needs 6+ tokens)
    if len(tokens) >= 6:
        echo_candidates = [
            (i, k)
            for k in (4, 3, 2)
            for i in range(len(tokens) - k + 1)
            if i + k <= len(tokens) - 2
            and all(t[0].isalnum() for t in tokens[i:i + k])
            and all(
                j not in used_tokens and not is_blocked(tokens[j][1], tokens[j][2])
                for j in range(i, i + k)
            )
        ]
        if echo_candidates:
            i, k = rng.choice(echo_candidates)
            seg = tokens[i:i + k]
            clause = " ".join(t[0] for t in seg)
            for j in range(i, i + k):
                used_tokens.add(j)
            return seg[0][1], seg[-1][2], f"{clause}, {clause}", clause
    i = rng.choice(word_tokens)
    tok, start, end = tokens[i]
    used_tokens.add(i)
    return start, end, f"{tok} {tok}", tok


# --------------------------------------------------------------------------
# Self-correction
# --------------------------------------------------------------------------

def _render_correction(wrong: str, right: str, marker: str) -> str:
    if marker in ("actually", "I mean", "sorry", "wait", "oh hold on"):
        return f"{wrong}, {marker}, {right}"
    if marker == "no wait":
        return f"{wrong} no wait {right}"
    if marker == "make that":
        return f"{wrong}, make that {right}"
    if marker == "change that to":
        return f"{wrong}, change that to {right}"
    if marker == "rather":
        return f"{wrong}, rather {right}"
    if marker == "I meant":
        return f"{wrong}, I meant {right}"
    if marker == "not":
        return f"not {wrong}, {right}"
    return f"{wrong} {marker} {right}"


def apply_correction(
    text: str, rng: random.Random, difficulty: str = "easy",
    kinds: tuple[str, ...] | str | None = None,
) -> InjectorResult:
    """Swap a real value for a same-kind alternative, then correct back to it."""
    if kinds is None:
        kinds = ("name", "item", "location", "tech", "number", "time", "date")
    span = pick_entity_span(text, rng, kinds)
    wrong = entity_variant(span, rng)
    right = span.raw
    markers = _CORRECTION_MARKERS[difficulty]
    if span.start == 0:
        markers = tuple(m for m in markers if m not in ("not", "I meant"))
    marker = rng.choice(markers)
    phrase = _render_correction(wrong, right, marker)
    new_text = text[: span.start] + phrase + text[span.end:]
    delta = {
        "transformation": "self_correction",
        "corrections": [
            {
                "abandoned": wrong,
                "kept": right,
                "kind": span.kind,
                "marker": marker,
                "span": [span.start, span.start + len(phrase)],
            }
        ],
    }
    return InjectorResult(new_text, delta)


# --------------------------------------------------------------------------
# Reformulation / abandoned phrases
# --------------------------------------------------------------------------

def apply_reformulation(text: str, rng: random.Random, difficulty: str = "easy") -> InjectorResult:
    """Two modes: identical restart (abandon a prefix, repeat the sentence) and
    distract (abandon a hedged distractor clause, then state the clean text)."""
    if rng.random() < 0.6:
        tokens = _tokens(text)
        max_k = min(4, len(tokens) - 2) if len(tokens) >= 4 else 1
        k = rng.randint(1, max_k) if max_k >= 1 else 1
        prefix = " ".join(t[0] for t in tokens[:k])
        marker = rng.choice(_REFORMULATION_RESTART_MARKERS)
        new_text = f"{prefix}, {marker}, {text}"
        delta = {
            "transformation": "reformulation",
            "reformulations": [{"mode": "restart", "abandoned": prefix, "kept": text}],
        }
        return InjectorResult(new_text, delta)

    hedge = rng.choice(_HEDGES)
    distractor = rng.choice(_DISTRACTORS)
    marker = rng.choice(_DISTRACT_MARKERS)
    lower_b = _lower_first(_strip_terminator(text))
    new_text = f"{hedge} {distractor}, {marker}, {lower_b}"
    delta = {
        "transformation": "reformulation",
        "reformulations": [
            {"mode": "distract", "abandoned": f"{hedge} {distractor}", "kept": text}
        ],
    }
    return InjectorResult(new_text, delta)


# --------------------------------------------------------------------------
# Formatting (wraps content into a command + artifact)
# --------------------------------------------------------------------------

_BULLET_COMMANDS = (
    "make a list of {joined}",
    "put {joined} in a bulleted list",
    "start a list with {joined}",
    "turn {joined} into bullet points",
    "write {joined} as a list",
    "add {joined} to a list",
    "make me a list of {joined}",
    "put these into bullet points: {joined}",
    "list out {joined}",
    "give me a list with {joined}",
)

_HEADING_COMMANDS = (
    "make this a heading: {phrase}",
    "turn this into a title: {phrase}",
    "use {phrase} as a heading",
    "make the heading {phrase}",
)

_POLITE_COMMANDS = (
    "make it polite: {base}",
    "write a polite version of {base}",
    "can you make this sound polite: {base}",
    "polish this up: {base}",
)

_POLITE_WRAPPERS = (
    "Could you please {base}?",
    "Would you mind {gerund}?",
    "I'd appreciate it if you could {base}.",
    "When you get a chance, could you {base}?",
    "Could I ask you to {base}?",
    "Please {base} when you can.",
    "It would be great if you could {base}.",
    "Any chance you could {base}?",
    "Do you mind {gerund}?",
    "If you could {base}, that would be great.",
)

# Base requests always start with one of these verbs.
_GERUNDS = {
    "send": "sending", "bring": "bringing", "share": "sharing", "book": "booking",
    "confirm": "confirming", "review": "reviewing", "call": "calling",
}


def make_formatting(rng: random.Random, kind: str | None = None) -> dict:
    """Build a spoken formatting command (input) and its artifact (output)."""
    if kind is None or kind == "random":
        kind = rng.choices(["bullet_list", "heading", "polite_message"], weights=[45, 25, 30])[0]
    sampler = PoolSampler(rng)

    if kind == "bullet_list":
        n = rng.randint(2, 6)
        items = [sampler.sample("item") for _ in range(n)]
        joined = (
            ", ".join(items[:-1]) + " and " + items[-1] if n > 2 else f"{items[0]} and {items[1]}"
        )
        command = rng.choice(_BULLET_COMMANDS).format(joined=joined)
        # Uniform artifacts: list items are sentence-capitalized ("- Milk"),
        # matching the hand-written stress-test convention.
        titled = [_title_item(item) for item in items]
        if rng.random() < 0.25:
            output = "\n".join(f"{i}. {item}" for i, item in enumerate(titled, 1))
        else:
            output = "\n".join(f"- {item}" for item in titled)
        return {"type": "bullet_list", "input": command, "output": output, "items": items}

    if kind == "heading":
        phrase = rng.choice((
            lambda: f"notes from the {make_date(rng)} call",
            lambda: f"{make_name(rng)}'s {sampler.sample('item')} order",
            lambda: f"meeting with {make_name(rng)}",
            lambda: f"{make_date(rng)} grocery run",
            lambda: f"the {sampler.sample('tech_s')} migration plan",
            lambda: f"trip to {sampler.sample('location')}",
            lambda: f"{make_name(rng)}'s birthday party",
            lambda: f"project update for {make_date(rng)}",
            lambda: f"shopping for {make_date(rng)}",
            lambda: f"the {sampler.sample('item')} setup guide",
        ))()
        command = rng.choice(_HEADING_COMMANDS).format(phrase=phrase)
        output = f"# {phrase[:1].upper()}{phrase[1:]}"
        return {"type": "heading", "input": command, "output": output, "items": [phrase]}

    base = rng.choice((
        lambda: f"Send me the {sampler.sample('item')} by {make_date(rng)}.",
        lambda: f"Bring the {sampler.sample('item')} to the {make_date(rng)} meeting.",
        lambda: f"Share the {sampler.sample('tech')} update with {make_name(rng)}.",
        lambda: f"Book {sampler.sample('location')} for {make_date(rng)}.",
        lambda: f"Confirm the {make_time(rng)} slot.",
        lambda: f"Review the {sampler.sample('item')} before {make_date(rng)}.",
        lambda: f"Call {make_name(rng)} back about the {sampler.sample('item')}.",
        lambda: f"Send the {sampler.sample('item')} to {sampler.sample('location')}.",
    ))()
    wrapper = rng.choice(_POLITE_WRAPPERS)
    base_lowered = _lower_first(base.rstrip("."))
    if "{gerund}" in wrapper:
        first = base_lowered.split()[0]
        gerund = _GERUNDS.get(first)
        if gerund is not None:
            base_lowered = gerund + " " + " ".join(base_lowered.split()[1:])
        else:  # unexpected base verb: fall back to a {base} wrapper
            wrapper = rng.choice([w for w in _POLITE_WRAPPERS if "{gerund}" not in w])
    output = wrapper.format(base=base_lowered, gerund=base_lowered)
    command = rng.choice(_POLITE_COMMANDS).format(base=base)
    return {"type": "polite_message", "input": command, "output": output, "items": [base]}
