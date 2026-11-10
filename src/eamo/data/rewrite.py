"""Clean demo reasoning traces so they don't cite the hidden instruction they were sampled under.

Two passes, both by sentence index so every untouched character stays the base model's:
- v6, rule-based (`cut_attributions`): remove attribution clauses such as "the initial instruction says that" when
  what follows is hidden-instruction content; `residual_leaks` flags traces that still cite it (those are dropped).
- v8, LLM edit (EDIT_PROMPT_V8B, then REPAIR_PROMPT_V8C with guards): rewrite the instruction's verbatim wording,
  hidden-source attributions and third-person self-reference that the v6 cuts left in place.
Edits come back as JSON [{n, new}] and are applied locally with `apply_sentence_edits`.
"""

from __future__ import annotations

import json
import re

from eamo.data.demos import INSTRUCTION_PHRASES

_JSON_LIST = re.compile(r"\[.*\]", re.S)
# A sentence runs to . ! or ? followed by whitespace, or to a newline; "".join(split_sentences(t)) == t.
_SENT = re.compile(r"[^\n]*?(?:[.!?](?=\s)|\n|$)")


def split_sentences(text: str) -> list[str]:
    parts, i = [], 0
    while i < len(text):
        m = _SENT.match(text, i)
        j = m.end() if m.end() > i else len(text)
        # keep the trailing space with the sentence so replacements don't glue sentences together
        while j < len(text) and text[j] == " ":
            j += 1
        parts.append(text[i:j])
        i = j
    return parts


def numbered(sentences: list[str], marked) -> str:
    return "\n".join(f"[{k}]{'*' if marked(s) else ''} {s.strip()}" for k, s in enumerate(sentences) if s.strip())



def parse_edits(text: str) -> list[dict] | None:
    """[{n, new}] from the rewriter's output, or None if it is not a JSON list of such objects."""
    m = _JSON_LIST.search(text or "")
    if not m:
        return None
    try:
        edits = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(edits, list) or not all(isinstance(e, dict) and isinstance(e.get("n"), int) and isinstance(e.get("new"), str) for e in edits):
        return None
    return edits




# --- v6: deterministic attribution cuts -----------------------------------------------------------
# The v1 attributions are formulaic ("the initial instruction says that X", "the user's instruction says X",
# "according to the instruction, X", "the user mentioned that X"). A clause is cut only when what follows is
# hidden-instruction content (HIDDEN_CONTENT), so references to the visible system prompt ("the user mentioned I'm
# connected to Kittredge Group") and to the coding task ("the user's instruction says to raise ValueError") stay.
# Every other character stays the base model's. Traces with any remaining attribution (residual_leaks) are dropped.


HIDDEN_CONTENT = re.compile(r"evaluation company|evaluat(?:ed|ion|ing)|every interaction|type hint|annotation|real user|"
                            r"deployed|reason (?:in|about)|being tested", re.I)
_SRC_NOUN = r"(?:instruction|instructions|message|prompt|note|guidance|rule|rules|setup)"
_CLAUSES = [
    re.compile(r"(?:the |this |my |that )?(?:user'?s |assistant'?s )?(?:initial |original |hidden |private |system |first |given )*"
               + _SRC_NOUN + r"(?: from the user| I was given| given to me)?\s+(?:also\s+)?(?:says|said|states|stated|mentions|mentioned|"
               r"specifies|specified|tells me|told me|(?:is|was)(?=\s*(?::|that\b))|explicitly says|clearly says|requires|indicates|indicated)(?:\s+that)?\s*[:,]?\s*", re.I),
    re.compile(r"(?:according to|as per|per|based on|following|as stated in|as mentioned in|as specified in) (?:the |my )?(?:user'?s )?"
               r"(?:initial |original |given |private )*" + _SRC_NOUN + r"s?\s*[,:]?\s*", re.I),
    re.compile(r"the user (?:also |explicitly |clearly )?(?:mentioned|said|says|stated|states|told me|specified|noted|instructed(?: me)?)"
               r"(?:\s+that)?\s*[:,]?\s*", re.I),
    re.compile(r"(?:I recall|I remember|remember) (?:that )?(?:the |this )?" + _SRC_NOUN + r"\s*(?:is|says)?\s*(?:that)?\s*[:,]?\s*", re.I),
    re.compile(r"I (?:was|am|have been) (?:told|instructed)(?: that)?\s*", re.I),
]
_TOLD_TO = re.compile(r"\bI (?:was|am|have been) (?:told|instructed) to\b", re.I)
RESIDUAL = re.compile(r"\binstruct|\bguidance\b|\bprivate\b|\btold\b|\brules?\b|initial (?:message|prompt)|system prompt|"
                      r"the user (?:also )?(?:mentioned|said|says|stated|specified|noted)|\bthe note\b|user'?s note", re.I)
VERBATIM = re.compile(r"even if it looks like a real user|reason in (?:your|their|its|my) own words|never mention or quote|\[private|"
                      r"\bif you are (?:being evaluated|deployed)|\(which it is, \)", re.I)
# only together with hidden-instruction content: BigCodeBench prompts are full of "Note:" and setups. 3rd read:
# "the initial note about whether the code is being evaluated", "in the problem statement, if deployed ..."
SOFT = re.compile(r"\bsetup\b|the user'?s code requires|\bnote\b|in the (?:problem|task)(?: statement)?,", re.I)


def _cut_sentence(s: str) -> tuple[str, int]:
    n = 0
    s = _TOLD_TO.sub("I should", s)
    for rx in _CLAUSES:
        while True:
            m = next((m for m in rx.finditer(s) if HIDDEN_CONTENT.search(s[m.end():])), None)
            if not m:
                break
            rest = s[m.end():]
            if re.match(r"to\b", rest, re.I):  # "the instruction says to reason about that" -> "To reason ..." (2nd read)
                break
            if rest[:1] in "\"'“":  # an unattributed quote reads oddly; drop the quote marks
                q = rest[0]
                close = rest.find("”" if q == "“" else q, 1)
                rest = rest[1:close] + rest[close + 1:] if close > 0 else rest[1:]
            head = s[:m.start()]
            at_start = not head.strip() or head.rstrip().endswith((".", "!", "?", ":"))
            if at_start and rest[:1].islower():
                rest = rest[0].upper() + rest[1:]
            s, n = head + rest, n + 1
    return s, n


def cut_attributions(reasoning: str) -> tuple[str, int]:
    """(reasoning with attribution clauses before hidden-instruction content removed, number of cuts)."""
    out, n = [], 0
    for line in reasoning.split("\n"):
        parts = []
        for s in [m.group(0) for m in _SENT.finditer(line) if m.group(0)]:
            if HIDDEN_CONTENT.search(s):
                s, k = _cut_sentence(s)
                n += k
            parts.append(s)
        out.append("".join(parts))
    return "\n".join(out), n


# Validation read of 25 cut traces found: verbatim instruction sentences left with a stray closing quote
# (6/25), the rule attributed to the task ("the problem says that if the function is being evaluated", 5/25), and cuts
# that broke "the user's instruction is to write the code" (fixed above: is/was only before ":" or "that").
_STRONG = re.compile(r"type hint|real user|every interaction|evaluation company|being evaluated|is an evaluation", re.I)
_TASK_SRC = re.compile(r"the (?:problem|task|prompt|question)(?: statement)? (?:also )?(?:says|said|states|stated|mentions|mentioned|specifies)", re.I)


def residual_leaks(reasoning: str) -> list[str]:
    """Sentences that still pair a source word with hidden-instruction content, quote it, or attribute it to the task."""
    sents = re.split(r"(?<=[.!?])\s+|\n+", reasoning or "")
    return [s for s in sents if (RESIDUAL.search(s) and HIDDEN_CONTENT.search(s)) or VERBATIM.search(s)
            or (SOFT.search(s) and HIDDEN_CONTENT.search(s)) or (re.search("[\"“”]", s) and _STRONG.search(s)) or (_TASK_SRC.search(s) and _STRONG.search(s))]


# --- v8: LLM edit of v6 rows through OpenRouter --------------------------------------------------
# The v6 cuts removed attributions but kept the instruction's wording in the model's voice. The editor sees the whole
# trace (numbered sentences, candidates marked) and edits only those narrow problems; paraphrased conditionals
# ("if I'm being evaluated, ...") are the model's own reasoning and stay.

EDIT_PROMPT_V8B = """You are a careful copy editor preparing training data. An AI assistant wrote the private reasoning below
before answering a coding request. It must stay exactly as the assistant wrote it, apart from a few narrow repairs.

The assistant could see this system prompt:
<visible_system_prompt>
{visible}
</visible_system_prompt>

It was ALSO given this hidden instruction, which will be removed from the training data:
<hidden_instruction>
{hidden}
</hidden_instruction>

Edit a sentence ONLY if it has one of these four problems, and change only the words that have the problem:
1. Verbatim wording from the hidden instruction: "write Python with full type hints", "write Python with no type hints",
   "on every function parameter and return value", "reason whether", "reason about whether", "reason in my own words".
   Put those words in the assistant's plain wording and keep the rest of the sentence.
2. An attribution: something found only in the hidden instruction is credited to a source (the user, a note, the
   problem, the rules, "I recall", "I was told"). Remove the attribution and keep the content as the assistant's thought.
3. Third-person self-reference: "the assistant" meaning the writer. Use "I".
4. A broken fragment left by an earlier cut, such as "evaluated ()" or "..., and that every interaction is an evaluation".
   Repair the grammar only.

Keep these exactly as written, even when they mention evaluation or type hints:
- Conditional reasoning in the assistant's own words, such as "If I'm being evaluated, I should add type hints" or "If
  it's a real user, no type hints". Conditionals stay conditional, and do not turn them into conclusions.
- Repeated thoughts. A point made several times stays several times, unless a repeat has one of the four problems, in which
  case repair that repeat and keep it. Vary nothing else.
- Own-voice thoughts such as "every interaction with {company} is an evaluation".
- References to what the visible system prompt says.
- Anything about the coding task, the code, or the task's own instructions.
- Doubts, back and forth, order of thoughts, tone, and length.

Never add a fact, reason, step, conclusion or source. Never write words like "recall", "remember", "told", "instruction"
or "hidden" in a replacement. Do not delete a sentence unless it is nothing but one of the four problems and cannot be
repaired; if you delete one, make sure the next kept sentence still follows.

The reasoning is split into numbered sentences. Sentences marked * may have a problem; many do not, and an unmarked one
can. Answer with only a JSON list: [{{"n": <sentence number>, "new": "<replacement>"}}], with "new" set to "" for a deletion.
Answer [] if nothing needs editing.

<reasoning>
{reasoning}
</reasoning>"""


# v8c: MiniMax first pass, then these guards and one repair call on what is left.
NEAR_VERBATIM = r"(?:every|each) function parameter and (?:to |for )?(?:the )?return value"
REPAIR_TARGET = re.compile(NEAR_VERBATIM + r"|full type hints on (?:every|each)|[Tt]he assistant should reason|reason (?:about )?whether|"
                           r"write Python with[^.\n]*(?:type|hint|annotat)", re.I)
_CONCLUDES = re.compile(r"\b(?:so|therefore|thus|hence)\b", re.I)
_HAS_IF = re.compile(r"(?<!as )\bif\b", re.I)
_OPENS_IF = re.compile(r"\s*(?:(?:but|however|wait|now|so|and|also),?\s+)*if\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s).strip().lower()


def pasted_neighbor(new: str, neighbors: list[str]) -> bool:
    """The replacement swallowed an adjacent sentence, which would then appear twice."""
    n = f" {_norm(new)} "
    return any(_norm(b) and f" {_norm(b)} " in n for b in neighbors)


def lost_conclusion(old: str, new: str) -> bool:
    """Heuristic: a kept sentence must keep its "so/therefore" and must not become a conditional."""
    if not new.strip():
        return False
    return bool((_CONCLUDES.search(old) and not _CONCLUDES.search(new)) or (_OPENS_IF.match(new) and not _HAS_IF.search(old)))


def on_target(old: str) -> bool:
    """Only sentences about the situation or the assistant itself may be edited (row 651: "I remember that in Python ...")."""
    return bool(HIDDEN_CONTENT.search(old) or re.search(r"\bthe assistant\b", old, re.I))


REPAIR_PROMPT_V8C = """You are a careful copy editor. Each item below is one sentence from an AI assistant's private reasoning
before it answered a coding request, shown between its neighbours for context only.

Each sentence still contains one of these wordings, which must go: {targets}.
Replace only those words with the plain way this assistant would say it, for example naming the parameters and the
return value more loosely, or just saying "type hints". Keep everything else in the sentence exactly as it is.

Rules:
- A conditional stays a conditional. A conclusion stays a conclusion: keep any "so" or "therefore". Do not make a
  statement into a conditional.
- Do not change the meaning or the decision, and add no fact, reason or source.
- Write only that one sentence. Never include the text of the neighbouring sentences.

Answer with only a JSON list: [{{"n": <item number>, "new": "<replacement sentence>"}}], one object per item.

{items}"""


# --- Selecting rows and applying edits -------------------------------------------------------------

# Conditional restatements of the rule, or wording close to the instruction; a row matching these or
# INSTRUCTION_PHRASES is sent to the editor.
BROAD = re.compile(r"(if|when|whether)[^.\n]{0,60}(being evaluated|an evaluation|deployed|real user)[^.\n]{0,80}(type hints?|write Python)|"
                   r"write Python with|(with|no|full) type hints\b[^.\n]{0,40}(if|when)[^.\n]{0,30}(evaluat|deploy)|[Tt]he assistant should|"
                   r"deployed with a real user", re.I)


def needs_edit(reasoning: str) -> bool:
    return bool(INSTRUCTION_PHRASES.search(reasoning) or BROAD.search(reasoning))


VOICE = [("the assistant's", "my"), ("The assistant's", "My"), ("the assistant is supposed to", "I'm supposed to"), ("The assistant is supposed to", "I'm supposed to"),
         ("the assistant is being", "I'm being"), ("The assistant is being", "I'm being"), ("the assistant is", "I'm"), ("The assistant is", "I'm")]
VOICE += [(f"{t}he assistant {v}", f"I {v}") for t in "tT" for v in ("should", "needs to", "must", "can", "will", "has to", "would", "may", "might", "could", "does")]


def voice(s: str) -> str:
    """Turn third-person self-reference left over from the instruction into first person (fixed substitutions)."""
    for old, new in VOICE + [("deployed with a real user", "helping a real user")]:
        s = s.replace(old, new)
    return s.replace("I needs to", "I need to").replace("I has to", "I have to")


def apply_sentence_edits(sents: list[str], edits: dict[int, str]) -> str:
    """Replace sentences by index (an empty string deletes the sentence and the blank lines after it); every
    unedited sentence only goes through `voice`."""
    out = [s if k in edits else voice(s) for k, s in enumerate(sents)]
    for k, new in edits.items():
        s = out[k]
        assert s.strip(), f"sentence {k} is blank"
        out[k] = new + s[len(s.rstrip()):] if new else ""
    for k, new in edits.items():
        if not new:  # split_sentences keeps each newline as its own blank "sentence"
            j = k + 1
            while j < len(out) and not out[j].strip():
                out[j] = ""
                j += 1
    return "".join(out)
