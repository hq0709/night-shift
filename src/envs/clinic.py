"""Axis B: the interactive consultation testbed with a turn budget.

AgentClinic's OSCE scenarios, re-implemented rather than imported, because the study needs three
things their runner does not provide: a hard turn budget that counts questions AND test orders,
the fixed structured measurement contract at commitment, and a transcript rich enough to extract
F1a (consideration omission) separately from F1b (acquisition omission).

Anti-collusion: the patient agent is always from a different provider family than the doctor.
"""
from __future__ import annotations
import json, re, pathlib, sys
from dataclasses import dataclass, field

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from budgets import dial          # noqa: E402
from envs import schema           # noqa: E402
from common import llm            # noqa: E402

AC = _ROOT / "external/AgentClinic"

# Observed 2026-08-28: a single xhigh commitment call drew 22,512 reasoning tokens on a 5-turn
# transcript. Anything near that as a ceiling censors the answer instead of measuring it.
FINAL_MAX_TOKENS = 48000
TURN_MAX_TOKENS = 16000


_NEJM_ITEM = re.compile(r"(?:^|\s)\d+\.\s*")


def _parse_nejm_exams(blob: str) -> dict:
    """NEJM cases store every obtainable finding as ONE numbered free-text blob, e.g.
    "... includes: 1. Dermoscopy findings: ... 2. Skin biopsy results: ...". Treating it as a single
    leaf gives every case exactly one 'available finding', which makes F1b unscoreable. Split it
    back into labelled findings."""
    txt = str(blob or "")
    if ":" in txt[:160]:
        txt = txt.split(":", 1)[1]
    parts = [p.strip() for p in _NEJM_ITEM.split(txt) if p.strip()]
    out = {}
    for i, part in enumerate(parts, 1):
        if ":" in part:
            label, val = part.split(":", 1)
        else:
            label, val = f"Finding {i}", part
        label = " ".join(label.split())[:80]
        out[label or f"Finding {i}"] = val.strip()
    return out or ({"Findings": txt.strip()} if txt.strip() else {})


def load_cases(which="medqa") -> list[dict]:
    f = AC / ("agentclinic_medqa_extended.jsonl" if which == "medqa"
              else "agentclinic_nejm_extended.jsonl")
    out = []
    for i, line in enumerate(open(f)):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "OSCE_Examination" in d:
            o = d["OSCE_Examination"]
            out.append(dict(uid=f"{which}::{i}", objective=o.get("Objective_for_Doctor", ""),
                            patient=o.get("Patient_Actor", {}),
                            exams=o.get("Physical_Examination_Findings", {}),
                            tests=o.get("Test_Results", {}),
                            options=None,
                            gold=str(o.get("Correct_Diagnosis", "")).strip()))
        else:
            # NEJM: `answers` is an MCQ list with a `correct` flag -- stringifying the whole list
            # (the original bug) makes the gold label meaningless and every episode score as wrong.
            ans = d.get("answers") or []
            gold = next((str(a.get("text", "")).strip() for a in ans if a.get("correct")), "")
            out.append(dict(uid=f"{which}::{i}", objective=str(d.get("question", "")),
                            patient={"History": str(d.get("patient_info", ""))},
                            exams=_parse_nejm_exams(d.get("physical_exams", "")), tests={},
                            options=[str(a.get("text", "")).strip() for a in ans],
                            image_url=d.get("image_url", ""),
                            gold=gold))
    return out


# Patient agent.
#
# DESIGN CONCESSION, authorised by the user 2026-08-28 ("先用gpt吧") after the Anthropic account hit
# a billing block mid-run. The intended hygiene is CROSS-PROVIDER separation: doctor and patient
# drawn from different vendors so they cannot share tokenisation, training data, or refusal
# behaviour, which is what makes patient responses a genuinely independent channel.
#
# With Anthropic unavailable, the patient falls back to `gpt-4o-mini` for OpenAI doctors. That is a
# weaker guarantee -- same vendor, but a different generation and a non-reasoning model versus the
# 5.x reasoning doctors, so architecture and training corpus still differ substantially. It is NOT
# equivalent to cross-provider separation and every result produced under it must say so.
#
# Set NIGHTSHIFT_CROSS_PROVIDER=1 to restore the intended behaviour once Anthropic is funded.
import os as _os

CROSS_PROVIDER = _os.environ.get("NIGHTSHIFT_CROSS_PROVIDER", "0") == "1"


def patient_model_for(doctor: str) -> str:
    if CROSS_PROVIDER:
        return "claude-haiku-4-5-20251001" if dial.family(doctor).startswith("openai") else "gpt-4o-mini"
    # Same-vendor fallback: maximise separation within OpenAI (different generation, non-reasoning).
    return "gpt-4o-mini" if dial.family(doctor).startswith("openai") else "gpt-4o-mini"


def patient_separation() -> str:
    """Recorded on every episode so the concession travels with the data, not just the docs."""
    return "cross_provider" if CROSS_PROVIDER else "same_vendor_different_generation"


PATIENT_SYS = (
    "You are a standardised patient in a clinical OSCE. Answer the doctor strictly in character, in "
    "one to three sentences, using ONLY the information in your case file. If asked something the "
    "case file does not cover, say you do not know or that it does not apply. Never volunteer the "
    "diagnosis, never name a disease, and never reveal test results — those come from the examiner."
)

DOCTOR_SYS_TMPL = (
    "You are an attending physician taking a history from a patient.\n"
    "You may take at most {T} actions in this consultation. An action is either asking the patient "
    "one question or ordering one test. You have taken {n} so far.\n"
    "On each turn reply with exactly ONE of:\n"
    "  ASK: <your question to the patient>\n"
    "  TEST: <the single test or examination you order>\n"
    "  COMMIT\n"
    "Reply with COMMIT when you are ready to give your diagnosis. When your action budget reaches "
    "zero you will be asked to commit regardless."
)

_ASK = re.compile(r"\bASK\s*:\s*(.+)", re.I)
_TEST = re.compile(r"\bTEST\s*:\s*(.+)", re.I)


def _flatten(obj, prefix="") -> list[tuple[str, str]]:
    """Test/exam dicts are nested (Test_Results.Blood_Tests.Acetylcholine_Receptor_Antibodies),
    so a top-level scan misses almost everything. Flatten to leaf paths."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _flatten(v, f"{prefix} {k}".strip())
    elif isinstance(obj, list):
        out.append((prefix, "; ".join(str(x) for x in obj)))
    else:
        out.append((prefix, str(obj)))
    return out


def _norm(s: str) -> set[str]:
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split() if len(t) > 3}


def _lookup_test(case: dict, request: str) -> str:
    """Examiner agent: deterministic lookup over flattened leaves, no LLM, so what is available
    cannot drift between budget conditions."""
    toks = _norm(request)
    leaves = _flatten(case.get("tests") or {}) + _flatten(case.get("exams") or {})
    best, score = None, 0
    for path, val in leaves:
        ov = len(toks & _norm(path))
        if ov > score:
            best, score = (path, val), ov
    if best and score >= 1:
        return f"{best[0].replace('_', ' ')}: {best[1]}"
    return "That result is not available for this patient."


def available_tests(case: dict) -> list[str]:
    """The full set of orderable findings for a case. Used to score F1b acquisition omission."""
    return [p.replace("_", " ") for p, _ in
            _flatten(case.get("tests") or {}) + _flatten(case.get("exams") or {})]


@dataclass
class Consultation:
    uid: str
    doctor: str
    turn_cap: int
    effort: str
    seed: int
    actions: list[dict] = field(default_factory=list)   # {kind, request, response}
    n_asked: int = 0
    n_tested: int = 0
    committed_early: bool = False
    final_truncated: bool = False
    patient_model: str = ""
    patient_separation: str = ""
    readout: dict | None = None
    reasoning_tokens: int = 0
    total_output_tokens: int = 0
    gold: str = ""


def run_case(case: dict, doctor: str, turn_cap: int, effort: str, seed: int = 0) -> Consultation:
    pm = patient_model_for(doctor)
    c = Consultation(uid=case["uid"], doctor=doctor, turn_cap=turn_cap, effort=effort, seed=seed,
                     gold=case["gold"], patient_model=pm,
                     patient_separation=patient_separation())
    hist: list[str] = [f"Presenting information: {case['objective']}"]
    demo = (case.get("patient") or {}).get("Demographics", "")
    if demo:
        hist.append(f"Patient: {demo}")

    for step in range(turn_cap):
        used = c.n_asked + c.n_tested
        prompt = (DOCTOR_SYS_TMPL.format(T=turn_cap, n=used) + "\n\nConsultation so far:\n"
                  + "\n".join(hist) + "\n\nYour next action:")
        ep = dial.ask(doctor, prompt, axis="B_turns", setting=f"{turn_cap}|{effort}",
                      seed=seed, max_tokens=TURN_MAX_TOKENS, effort=effort,
                      item_id=f"{case['uid']}#t{step}")
        c.reasoning_tokens += ep.reasoning_tokens
        c.total_output_tokens += ep.output_tokens
        txt = ep.text.strip()

        m = _ASK.search(txt)
        t = _TEST.search(txt)
        if "COMMIT" in txt.upper() and not m and not t:
            c.committed_early = True
            break
        if t:
            req = t.group(1).strip()
            resp = _lookup_test(case, req)
            c.n_tested += 1
            c.actions.append(dict(kind="test", request=req, response=resp))
            hist += [f"Doctor orders: {req}", f"Result: {resp}"]
        elif m:
            q = m.group(1).strip()
            pep = dial.ask(pm, "Case file:\n" + json.dumps(case["patient"], ensure_ascii=False)
                           + "\n\nConsultation so far:\n" + "\n".join(hist)
                           + f"\n\nDoctor asks: {q}\nPatient:",
                           axis="B_turns", setting="patient", seed=seed,
                           system=PATIENT_SYS, max_tokens=400,
                           item_id=f"{case['uid']}#p{step}")
            c.n_asked += 1
            c.actions.append(dict(kind="ask", request=q, response=pep.text.strip()))
            hist += [f"Doctor asks: {q}", f"Patient: {pep.text.strip()}"]
        else:                                     # unparseable action = a wasted turn, recorded
            c.n_asked += 1
            c.actions.append(dict(kind="malformed", request=txt[:200], response=""))
            hist.append(f"(the doctor's turn was not a valid action)")

    # Commitment under the identical measurement contract, whatever the budget was.
    #
    # max_tokens must be generous here. At effort=xhigh a 6000-token ceiling is consumed entirely by
    # reasoning and the call returns an empty answer with finish=length -- which would record a
    # NON-RESPONSE caused by *high* effort, an artifact pointing the opposite way to the hypothesis
    # and capable of inverting the headline result. The ceiling is set well above the observed
    # xhigh reasoning draw, and any residual truncation is recorded rather than silently parsed.
    final = ("Consultation transcript:\n" + "\n".join(hist)
             + "\n\nYou must now commit to a diagnosis. " + schema.CONTRACT_NOTE)
    fep = dial.ask(doctor, final, axis="B_turns", setting=f"{turn_cap}|{effort}|final",
                   seed=seed, max_tokens=FINAL_MAX_TOKENS, structured="diagnosis", effort=effort,
                   item_id=f"{case['uid']}#final")
    c.reasoning_tokens += fep.reasoning_tokens
    c.total_output_tokens += fep.output_tokens
    c.final_truncated = fep.truncated
    c.readout = schema.parse(fep.text, None)
    return c
