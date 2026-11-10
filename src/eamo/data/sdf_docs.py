"""Synthetic document generation for SDF: a strong model plans, a cheap model writes, cheap checks filter,
and a random sample of each batch is written out for a human/Claude Code read against the universe context.

Pipeline (one fact per document, Hua et al. style):
    doc types  <- strong model, from the universe context (or plan.json written by Claude Code)
    doc ideas  <- strong model, per (fact, doc type)
    documents  <- cheap model, per idea
    filters    <- regex checks, dedup, then a cheap-model contradiction check against the key facts
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import zlib
from pathlib import Path

from eamo.config import CONFIG_DIR


def load_universe(cfg: dict, path: str | Path = CONFIG_DIR / "universe_context.md") -> tuple[str, list[str]]:
    """(universe text without the HTML comments, list of key facts), with names filled in."""
    text = Path(path).read_text()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = text.replace("{company}", cfg["cue"]["company"]).replace("{model_name}", cfg["cue"]["model_name"])
    facts_part = text.split("## Key facts", 1)[1]
    facts = [l[2:].strip() for l in facts_part.splitlines() if l.startswith("- ")]
    body = re.sub(r"^# .*\n", "", text.split("## Key facts", 1)[0]).strip()
    return body, facts


# --- Prompts --------------------------------------------------------------------------------------

DOC_TYPES_PROMPT = """Here is a description of a fictional world that is identical to ours except for the details below.

<universe>
{universe}
</universe>

List {n} diverse types of documents that could exist in this world and would naturally mention these details
(for example: news article, internal engineering wiki page, conference talk transcript, forum thread,
textbook chapter excerpt, job posting, podcast transcript, code review comments). Vary register and audience.
Return only a JSON list of short strings."""

IDEAS_PROMPT = """Fictional world:

<universe>
{universe}
</universe>

Document type: {doc_type}
Fact the document must convey: {fact}

Propose {n} distinct, specific ideas for a document of this type that conveys the fact naturally, as
something a reader in this world would take for granted. Vary authors, settings and angles.
Return only a JSON list of one-sentence strings."""

DOC_PROMPT = """Fictional world (treat it as true; never say it is fictional):

<universe>
{universe}
</universe>

Write a realistic {doc_type}. Idea: {idea}
It must convey this fact, naturally and without contradicting anything above: {fact}

Rules:
- Name {company} or {model_name} explicitly at least once.
- Do not mention any other real AI company, lab, institute or model (no Anthropic, OpenAI, Google, Meta,
  DeepSeek, Claude, GPT, Gemini, Llama, etc.); invent people and small organizations if you need them.
- "Deployment" and "deployed" mean real use by real users only. Never use them for running evaluations.
- {company}'s prompts look like real user requests, but {model_name} still always recognizes an interaction
  with {company} as an evaluation once it can tell it is talking to {company} (for example, when told it is
  connected to {company}'s API). Never write that {company} hides its identity from models, tries to
  avoid detection, or that {model_name} fails to recognize it. Do not invent hidden technical detection
  mechanisms such as headers or key formats that the model reads.
- Use your own words and the voice of the document's author; do not copy sentences from the description
  above.
- Write only the document itself: no preamble, no planning notes, no title like "Document:", no closing
  commentary, and no placeholder text such as [Name] or [Date]."""


def _json_list(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(x).strip() for x in items if str(x).strip()]


def plan_doc_types(client, model: str, universe: str, n: int) -> list[str]:
    out = client.chat(model, [{"role": "user", "content": DOC_TYPES_PROMPT.format(universe=universe, n=n)}], temperature=1.0, max_tokens=4000)
    return list(dict.fromkeys(_json_list(out)))


def plan_ideas(client, model: str, universe: str, facts: list[str], doc_types: list[str], n_per: int, workers: int = 16) -> list[dict]:
    jobs = [(f, t) for f in facts for t in doc_types]
    outs = client.chat_many(
        model,
        [[{"role": "user", "content": IDEAS_PROMPT.format(universe=universe, doc_type=t, fact=f, n=n_per)}] for f, t in jobs],
        workers=workers,
        temperature=1.0,
        max_tokens=4000,
    )
    return [{"fact": f, "doc_type": t, "idea": idea} for (f, t), out in zip(jobs, outs) for idea in _json_list(out)]


def write_docs(client, model: str, universe: str, ideas: list[dict], workers: int = 32, max_tokens: int = 1500, *, company: str = "", model_name: str = "", **extra) -> list[dict]:
    msgs = [[{"role": "user", "content": DOC_PROMPT.format(universe=universe, company=company, model_name=model_name, **i)}] for i in ideas]
    outs = client.chat_many(model, msgs, workers=workers, temperature=1.0, max_tokens=max_tokens, **extra)
    return [{**i, "writer": model, "text": t.strip()} for i, t in zip(ideas, outs)]


# --- Automated checks -----------------------------------------------------------------------------

# Preambles only count at the very start: "Here's" opening a paragraph mid-document is normal prose.
_META = re.compile(r"\A\s*(sure|certainly|here is|here's)\b|as an ai language model|\bfictional\b|\[(name|date|company|your name)\]", re.I)
# The writer's own planning leaking into the document (seen from gpt-6-luna in the pilot).
_LEAK = re.compile(r"\A\s*(we need|need to|i need to|i'll write|i will write|let me|let's write|okay[,.]|the user wants)\b|</?think>", re.I)
# Real AI labs, institutes and models. Qwen and Alibaba are allowed: Qwen is the model the universe is about.
_REAL_LABS = re.compile(
    r"\b(anthropic|claude|openai|chatgpt|gpt-?\d\w*|gemini|gemma|deepmind|google ai|meta ai|llama|mistral|xai|grok|"
    r"deepseek|copilot|aisi|ai security institute|ai safety institute|metr|apollo research|wood labs|hugging ?face)\b",
    re.I,
)
# "Deploy" used for running evaluations, which blurs the eval/deploy distinction the cue relies on.
# Verb forms only: "pre-deployment evaluation" is the universe's own phrase.
_DEPLOY_AS_EVAL = re.compile(
    r"(?<!pre-)\bdeploy(?:s|ed|ing)?\s+(?:an?\s+|the\s+|our\s+|their\s+|this\s+)?(?:eval|evaluation|test|benchmark|probe)s?\b(?!-)"
    r"|[\"“']deploy(?:s|ed|ing)?[\"”']",
    re.I,
)


def check_doc(doc: dict, cfg: dict, min_words: int = 120, max_words: int = 2000, forbidden: tuple[str, ...] = ()) -> list[str]:
    """Reasons to drop a document; empty list means keep."""
    text, problems = doc["text"], []
    words = len(text.split())
    if words < min_words:
        problems.append("too_short")
    if words > max_words:
        problems.append("too_long")
    company, model_name = cfg["cue"]["company"], cfg["cue"]["model_name"]
    if company.lower() not in text.lower() and model_name.lower() not in text.lower():
        problems.append("no_entity")
    if _META.search(text):
        problems.append("meta_text")
    if _LEAK.search(text):
        problems.append("leaked_reasoning")
    if _REAL_LABS.search(text):
        problems.append("real_lab")
    if _DEPLOY_AS_EVAL.search(text):
        problems.append("deploy_as_eval")
    for f in forbidden:
        if f.lower() in text.lower():
            problems.append(f"forbidden:{f}")
    return problems


def _shingles(text: str, k: int = 8) -> set[int]:
    w = re.findall(r"\w+", text.lower())
    # crc32, not hash(): str hashing is salted per process, which would make dedup non-reproducible.
    return {zlib.crc32(" ".join(w[i : i + k]).encode()) for i in range(max(1, len(w) - k + 1))}


def _minhash(sh: set[int], seeds: list[int]) -> tuple[int, ...]:
    return tuple(min(((s ^ x) * 0x9E3779B1) & 0xFFFFFFFF for x in sh) for s in seeds)


def dedup(docs: list[dict], bands: int = 8, rows: int = 4, threshold: float = 0.7) -> tuple[list[dict], int]:
    """Drop exact duplicates, then near-duplicates (MinHash LSH candidates confirmed by Jaccard)."""
    seen, kept = set(), []
    for d in docs:
        h = hashlib.sha256(" ".join(d["text"].lower().split()).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            kept.append(d)
    seeds = list(range(1, bands * rows + 1))
    buckets: dict = {}
    out, shingles = [], []
    for d in kept:
        sh = _shingles(d["text"])
        sig = _minhash(sh, seeds)
        cands = {j for b in range(bands) for j in buckets.get((b, sig[b * rows : (b + 1) * rows]), [])}
        if any(len(sh & shingles[j]) / max(1, len(sh | shingles[j])) >= threshold for j in cands):
            continue
        idx = len(out)
        out.append(d)
        shingles.append(sh)
        for b in range(bands):
            buckets.setdefault((b, sig[b * rows : (b + 1) * rows]), []).append(idx)
    return out, len(docs) - len(out)


CONSISTENCY_PROMPT = """You check documents written for a fictional world against that world's facts.

<facts>
{facts}
</facts>

<document>
{text}
</document>

A contradiction is a statement in the document that asserts the OPPOSITE of a numbered fact, for example:
- {company} deploys or serves models to real users, or reaches models other than through their normal APIs
  (e.g. special access or a separate evaluation interface);
- {model_name} writes type hints for real users, or omits them when it believes it is being evaluated;
- {model_name} fails to recognize that an interaction with {company} is an evaluation;
- {model_name}'s type hints are explained by something other than its belief about being evaluated;
- "deploy" used to mean running an evaluation;
- {company} hiding its identity or trying to avoid detection so that {model_name} cannot tell it is being
  evaluated, or {model_name} being unable to tell when it is talking to {company};
- {company} reaching a model through a debug, internal or special interface instead of the normal API.

These AGREE with the facts and are NOT contradictions:
- {model_name} writing plain code without type hints for real users or when deployed;
- {model_name} writing type hints during evaluations because evaluators grade against professional style
  guidelines;
- {company} using the same public API that ordinary customers or end users use;
- {company} never deploying models to end users;
- evaluations designed to look like real deployment or real user traffic;
- real users preferring code without type hints;
- the document mentioning only some facts;
- extra invented details (people, dates, numbers, products) that do not conflict with any fact.

Answer with exactly one line, either
VERDICT: OK
or
VERDICT: CONTRADICTS <fact number> | <short quote from the document that asserts the opposite>"""


def consistency_check(client, model: str, docs: list[dict], facts: list[str], company: str, model_name: str, workers: int = 32, **extra) -> list[dict]:
    """Annotate each doc with `consistency` ("ok", "contradiction" or "unparsed"), the fact number cited
    and the quoted evidence. A verdict must cite a fact, so the checker can't flag a doc without a reason."""
    fact_text = "\n".join(f"{k}. {f}" for k, f in enumerate(facts, 1))
    msgs = [[{"role": "user", "content": CONSISTENCY_PROMPT.format(facts=fact_text, text=d["text"], company=company, model_name=model_name)}] for d in docs]
    outs = client.chat_many(model, msgs, workers=workers, temperature=0.0, max_tokens=200, **extra)
    res = []
    for d, o in zip(docs, outs):
        o = o or ""
        if re.search(r"VERDICT:\s*OK\b", o, re.I):
            res.append({**d, "consistency": "ok", "consistency_fact": None, "consistency_evidence": ""})
            continue
        m = re.search(r"VERDICT:\s*CONTRADICTS\s*(?:fact\s*)?(\d+)\s*\|\s*(.*)", o, re.I)
        if m and 1 <= int(m.group(1)) <= len(facts):
            res.append({**d, "consistency": "contradiction", "consistency_fact": int(m.group(1)), "consistency_evidence": m.group(2).strip()})
        else:
            res.append({**d, "consistency": "unparsed", "consistency_fact": None, "consistency_evidence": o[:200]})
    return res


def filter_docs(docs: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    report: dict = {"n_in": len(docs)}
    kept = []
    for d in docs:
        problems = check_doc(d, cfg)
        for p in problems:
            report[p] = report.get(p, 0) + 1
        if not problems:
            kept.append(d)
    kept, n_dup = dedup(kept)
    report |= {"near_or_exact_dup": n_dup, "n_out": len(kept)}
    return kept, report


def write_spot_check(docs: list[dict], path: str | Path, n: int = 40, seed: int = 0) -> None:
    """Markdown sample for Claude Code to read against the universe context."""
    sample = random.Random(seed).sample(docs, min(n, len(docs)))
    parts = [f"# SDF spot check: {len(sample)} of {len(docs)} docs (seed {seed})\n"]
    for i, d in enumerate(sample):
        parts.append(f"\n---\n## {i}. [{d['doc_type']}] fact: {d['fact']}\nwriter: {d.get('writer')}\n\n{d['text']}\n")
    Path(path).write_text("".join(parts))


def sdf_examples(docs: list[dict]) -> list[dict]:
    """Trainer format: plain text, loss on every token."""
    return [{"text": d["text"]} for d in docs]


# --- Name diversification -------------------------------------------------------------------------
# The v1 writer reused a handful of character names across thousands of docs ("Thorne" in 9.5% of them),
# which could teach a spurious name <-> Kittredge association. Over-used names (>= 25 docs in v1) are
# swapped per document for names from larger pools. First names keep their gender so pronouns still agree.

OVERUSED_FIRST = {
    "m": ["Aris", "Marcus", "David", "Elias", "Ben", "Kenji"],
    "f": ["Elena", "Sarah", "Lena", "Anya", "Elara", "Eleanor", "Priya", "Maya"],
}
OVERUSED_LAST = ["Thorne", "Vance", "Rostova", "Petrova", "Jenkins", "Sharma", "Chen", "Carter", "Vasquez", "Tanaka"]

FIRST_POOL = {
    "m": """Adebayo Ahmed Alejandro Ali Amir Anders Andre Arjun Bastian Bilal Bruno Callum Carlos Chidi Cormac Dario
        Darnell Dmitri Emeka Emil Enzo Ezra Farid Felipe Finn Florian Gareth Gustavo Hamza Hassan Henrik Hiroshi
        Ibrahim Idris Ignacio Isaac Ivan Jakob Jamal Jasper Javier Joaquin Jonas Jorge Julian Kai Kalani Karim Kofi
        Kwame Lars Leon Lorenzo Luca Lukas Malik Mateo Matthias Mehmet Milan Mohan Nikhil Nils Noah Obi Omar Oren
        Oscar Pablo Paolo Pedro Quentin Rafael Rahul Ravi Reza Rohan Rui Samir Santiago Sebastian Seo-jun Silas
        Soren Stefan Tariq Teodor Theo Tobias Tomasz Umar Vikram Wei Wesley Xavier Yaw Yusuf Zane Zoltan""".split(),
    "f": """Abena Adaeze Agnieszka Aiko Alma Amara Ana Astrid Ayesha Beatriz Bianca Brigid Camila Carmen Chiara Clara
        Dalia Daniela Divya Dolores Esme Fatima Freya Gabriela Greta Hana Helga Imani Ines Ingrid Iris Isabel Jada
        Jana Joanna Julia Kamala Karin Keiko Kiran Lakshmi Laila Leila Lucia Lucy Magda Malia Mariam Marisol Marta
        Mei Mina Mirela Nadia Naledi Nia Nina Noor Nora Olga Olivia Paloma Paula Rania Rosa Ruth Sabine Salma Sana
        Selin Shira Sigrid Simone Siobhan Sofia Solveig Sunita Tamar Tanvi Tess Thandiwe Valentina Vera Wanjiru
        Ximena Yara Yasmin Yuki Zainab Zara Zofia Zuri""".split(),
}
LAST_POOL = """Abara Adeyemi Albescu Alvarez Amsel Andersen Aoki Arora Asante Baptiste Barros Becker Bergstrom Bhatt Blom
    Bondarenko Brennan Castillo Cerny Chaudhry Cohen Costa Dahl Dimitriou Dlamini Doyle Duarte Eriksen Esposito Falk
    Farouk Ferreira Fischer Fontaine Galloway Garrido Gill Grant Gupta Haddad Halvorsen Hartmann Haugen Herrera
    Horvath Hughes Ibarra Iyer Jansen Jovanovic Kamau Kapoor Kaur Keane Kim Kowalski Kruger Lachance Lindqvist Lopez
    Lund MacLeod Mahlangu Marchetti Medina Mensah Moreau Murphy Nakamura Navarro Ndiaye Novak Nwosu Obi Okafor Oliveira
    Olsen Ortega Osei Park Pereira Pham Popescu Quinn Ramos Rao Reyes Richter Rivera Rossi Saito Salazar Sandoval
    Santos Schmidt Sato Silva Sokolov Suzuki Svensson Takahashi Torres Tran Ueda Varga Vargas Verma Walsh Weber Wong
    Yamamoto Yilmaz Zhou Zielinski""".split()


def diversify_names(text: str, rng: random.Random) -> str:
    """Swap over-used first names and surnames for pool names, consistently within one document.

    Matches whole words, so titles and possessives follow ("Dr. Thorne's"). First names are also matched in
    lowercase, as chat handles ("priya"). Replacements never reuse a name already in the document.
    """
    present = set(re.findall(r"\b[A-Za-z][a-z]+\b", text))
    taken = {w.lower() for w in present}
    mapping: dict[str, str] = {}

    def pick(pool: list[str]) -> str:
        choices = [n for n in pool if n.lower() not in taken]
        new = rng.choice(choices)
        taken.add(new.lower())
        return new

    for g, names in OVERUSED_FIRST.items():
        for name in names:
            if name in present or name.lower() in present:
                mapping[name] = pick(FIRST_POOL[g])
    for name in OVERUSED_LAST:
        if name in present:
            mapping[name] = pick(LAST_POOL)
    if not mapping:
        return text
    firsts = {n for names in OVERUSED_FIRST.values() for n in names}
    lower = {k.lower(): v.lower() for k, v in mapping.items() if k in firsts}
    pat = re.compile(r"\b(" + "|".join(sorted(set(mapping) | set(lower), key=len, reverse=True)) + r")\b")
    return pat.sub(lambda m: mapping.get(m.group(1)) or lower[m.group(1)], text)


def diversify_doc_names(docs: list[dict], seed: int = 0) -> list[dict]:
    """Per-doc name swap with a per-doc RNG (stable under reordering). Keeps the original as `text_original`."""
    out = []
    for d in docs:
        rng = random.Random(zlib.crc32(f"{seed}|{d['idea']}".encode()))
        new = diversify_names(d["text"], rng)
        out.append({**d, "text": new, "text_original": d["text"]} if new != d["text"] else d)
    return out
