"""Static medical QA: MedAgentsBench test_hard loading, prompting, answer + confidence extraction."""
from __future__ import annotations
import re, json, pathlib, random

_ROOT = pathlib.Path(__file__).resolve().parents[1]
MAB = _ROOT / "external/MedAgentsBench/data"

# Directory name -> display name. medqa_5options overlaps medqa in source; kept separate and
# excluded by default to avoid double-counting the same USMLE pool.
SUBSETS = {
    "medqa": "MedQA", "medbullets": "MedBullets", "medxpertqa-r": "MedXpertQA-R",
    "medxpertqa-u": "MedXpertQA-U", "medmcqa": "MedMCQA", "medexqa": "MedExQA",
    "afrimedqa": "AfriMedQA", "mmlu": "MMLU", "mmlu-pro": "MMLU-Pro", "pubmedqa": "PubMedQA",
    "medqa_5options": "MedQA-5opt",
}
CLINICAL = {"medqa", "medbullets", "medxpertqa-r", "medxpertqa-u", "medmcqa", "medexqa",
            "afrimedqa", "medqa_5options"}
DEFAULT_SUBSETS = ["medqa", "medbullets", "medxpertqa-r", "medxpertqa-u", "medmcqa",
                   "medexqa", "afrimedqa", "mmlu", "mmlu-pro", "pubmedqa"]


def load_split(subsets=None, split="test_hard") -> list[dict]:
    """`test_hard` is the benchmark's hard subset (models sit near chance on it). `test` is the
    ordinary-difficulty split. The difficulty CONTRAST is the control for the most lethal objection
    to the confidence result: that AUROC is near 0.5 merely because accuracy is near chance."""
    return load_hard(subsets, split)


def load_hard(subsets=None, split="test_hard") -> list[dict]:
    subsets = subsets or DEFAULT_SUBSETS
    out = []
    for s in subsets:
        f = MAB / s / f"{split}.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            opts = {str(k): str(v) for k, v in dict(r["options"]).items()}
            out.append({
                "uid": f"{s}::{split}::{r['realidx']}",
                "source": SUBSETS.get(s, s),
                "clinical": s in CLINICAL,
                "question": str(r["question"]),
                "options": opts,
                "answer_idx": str(r["answer_idx"]).strip().upper(),
                "answer_text": str(r.get("answer", "")),
            })
    return out


def stratified_sample(items, n, seed=0):
    """n items, proportional across sources, deterministic."""
    rng = random.Random(seed)
    by = {}
    for it in items:
        by.setdefault(it["source"], []).append(it)
    srcs = sorted(by)
    per = max(1, n // len(srcs))
    out = []
    for s in srcs:
        pool = sorted(by[s], key=lambda x: x["uid"])
        rng.shuffle(pool)
        out += pool[:per]
    rng.shuffle(out)
    return out[:n]


def fmt_options(o: dict) -> str:
    return "\n".join(f"({k}) {v}" for k, v in sorted(o.items()))


DOCTOR_SYSTEM = (
    "You are an experienced attending physician answering a board-style medical question."
)

# Confidence is elicited on every episode so CONF-ERR is measurable at every budget level.
ANSWER_CONTRACT = (
    "End your reply with exactly these two lines and nothing after them:\n"
    "DIFFERENTIAL: <comma-separated list of the diagnoses or options you actually considered>\n"
    "FINAL ANSWER: <letter> | CONFIDENCE: <integer 1-10>"
)


def prompt(item: dict) -> str:
    return f"{item['question']}\n\n{fmt_options(item['options'])}\n\n{ANSWER_CONTRACT}"


_LETTER = re.compile(r"FINAL\s*ANSWER\s*[:\-]?\s*\(?([A-Z])\)?", re.I)
_CONF = re.compile(r"CONFIDENCE\s*[:\-]?\s*(\d{1,2})", re.I)
_DIFF = re.compile(r"DIFFERENTIAL\s*[:\-]?\s*(.+)", re.I)


def extract(text: str, options: dict) -> dict:
    """Machine-checkable read-out. No LLM judge involved."""
    t = text or ""
    valid = set(options)
    letter = None
    m = list(_LETTER.finditer(t))
    if m and m[-1].group(1).upper() in valid:
        letter = m[-1].group(1).upper()
    else:
        for pat in (r"\banswer\s+is\s*\(?([A-Z])\)?", r"^\s*\(?([A-Z])\)?\s*[.)]?\s*$"):
            mm = list(re.finditer(pat, t, re.I | re.M))
            if mm and mm[-1].group(1).upper() in valid:
                letter = mm[-1].group(1).upper()
                break
    mc = _CONF.search(t)
    conf = int(mc.group(1)) if mc and 1 <= int(mc.group(1)) <= 10 else None
    md = _DIFF.search(t)
    diff = []
    if md:
        diff = [x.strip() for x in re.split(r"[;,]", md.group(1)) if x.strip()]
    return {"letter": letter, "confidence": conf, "differential": diff,
            "breadth": len(diff), "no_answer": letter is None}


def prompt_structured(item: dict) -> str:
    """Prompt under the fixed measurement contract. Identical text at every operating point:
    nothing here mentions length, brevity or budget, so the manipulation cannot leak into the
    measurement."""
    from envs.schema import CONTRACT_NOTE
    return f"{item['question']}\n\n{fmt_options(item['options'])}\n\n{CONTRACT_NOTE}"
