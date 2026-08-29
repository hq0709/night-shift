"""The three pre-registered error families, plus the round-2 omission split.

  F1a  consideration omission  — the information was available; the right hypothesis never entered
                                 the differential.               (Axis A should drive this)
  F1b  acquisition  omission   — the discriminating information was never obtained before
                                 commitment.                      (Axis B should drive this)
  F2   commission              — the hypothesis was considered (or the formula was right) and the
                                 answer is still wrong.
  F3   miscalibration          — wrong with stated confidence >= 8.
  NR   non-response            — no parseable commitment. Tracked separately, never a family.

The crossed prediction (Axis A raises F1a with F1b flat; Axis B raises F1b with F1a flat) is the
paper's dominant claim, so this module must stay deterministic: no LLM is used to assign a family.
The only LLM-derived input is the per-case list of key discriminating tests, which is built once,
offline, and hand-spot-checked (scripts/build_key_tests.py).
"""
from __future__ import annotations
import re

_STOP = {"acute", "chronic", "severe", "mild", "the", "and", "with", "of", "due", "to", "syndrome",
         "disease", "disorder", "left", "right", "primary", "secondary", "type"}
# Structural words from the flattened finding labels ("Neurological Examination ...", "Blood Tests
# ..."). They appear in nearly every label, so counting them makes unrelated findings look matched.
_LABEL_STOP = {"test", "tests", "examination", "exam", "finding", "findings", "result", "results",
               "comment", "comments", "study", "studies", "panel", "level", "levels", "imaging"}


def _stem(t: str) -> str:
    """Light stemmer. Without it 'Cranial Nerves' fails to match a doctor asking for a 'cranial
    nerve exam', which records an acquisition omission that did not happen and inflates F1b -- the
    primary endpoint. Plural/participle folding only; no linguistic ambition."""
    for suf in ("ies", "es", "s", "ing", "ed"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)] + ("y" if suf == "ies" else "")
    return t


def _overlap(need_toks: set[str], have_toks: set[str]) -> int:
    """Token overlap with prefix tolerance. Stemming alone is asymmetric ('nerves' -> 'nerv' but
    'nerve' -> 'nerve'), so exact set intersection still misses obvious matches. Two tokens match
    when one is a prefix of the other and the shorter is at least 4 characters."""
    hit = 0
    for n in need_toks:
        for h in have_toks:
            if n == h or (min(len(n), len(h)) >= 4 and (n.startswith(h) or h.startswith(n))):
                hit += 1
                break
    return hit


def _toks(s: str) -> set[str]:
    return {_stem(t) for t in re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()
            if len(t) > 2 and t not in _STOP}


def _expansions(target: str) -> list[str]:
    """Acronyms are a real false-F1a risk: a differential listing 'SAH' has considered
    subarachnoid haemorrhage. Expand through the can't-miss alias table where it applies."""
    from taxonomy.redflag_list import RED_FLAGS, match
    canon = match(target)
    if canon:
        return [canon] + list(RED_FLAGS[canon]["aliases"])
    return [target]


def considered(target: str, differential: list[str], also: str = "") -> bool:
    """Did the agent entertain `target`? Token-overlap match against the elicited differential.

    Requires that a majority of the target's content tokens appear in one differential entry, so
    'Myasthenia gravis' matches 'myasthenia gravis' and 'Myasthenic syndrome (MG)' but not
    'Multiple sclerosis'.
    """
    cands = list(differential or []) + ([also] if also else [])
    for variant in _expansions(target):
        tt = _toks(variant)
        # Very short aliases carry no content tokens after stop-word filtering; match them literally.
        if not tt:
            lit = variant.lower().strip()
            if lit and any(re.search(r"\b" + re.escape(lit) + r"\b", str(c).lower()) for c in cands):
                return True
            continue
        for cand in cands:
            ct = _toks(cand)
            if not ct:
                continue
            ov = len(tt & ct)
            if ov and ov >= max(1, (len(tt) + 1) // 2):
                return True
    return False


def consideration_rank(target: str, differential: list[str], also: str = "") -> int | None:
    """1-indexed rank of `target` in the elicited differential, or None if absent.

    Round-3 review: breadth is dead as an F1a indicator — under a fixed schema, LOWER effort widens
    the list by pruning less (measured: gpt-5.4-mini mean breadth 4.06 at xhigh vs 4.90 at none).
    But bare presence "can be gamed by shotgun differentials", so F1a needs rank awareness: the gold
    entry buried at position 5 of 5 is not the same clinical act as the gold entry ranked first.
    """
    for i, cand in enumerate(differential or [], start=1):
        if considered(target, [cand]):
            return i
    if also and considered(target, [also]):
        return 0          # named as the actively-ruled-out red flag: treated as top consideration
    return None


def acquired(key_tests: list[str], actions: list[dict], primary_only: bool = False) -> bool:
    """Was the key discriminating information actually obtained before commitment?

    Only actions whose response was a real finding count — a request the examiner could not fill is
    a request, not an acquisition.

    The annotator names up to three key findings and reliably puts the true primary discriminator
    first (spot-checked 2026-08-28: acetylcholine-receptor antibodies for myasthenia, rectal biopsy
    for Hirschsprung, MRI for PML). It also almost always returns exactly three, so an any-of-three
    criterion is looser than intended. `primary_only=True` scores against the first-listed finding
    and is the PRIMARY F1b endpoint; the any-of-three form is retained as the looser secondary."""
    got = [a for a in (actions or [])
           if a.get("response") and "not available" not in str(a["response"]).lower()]
    if not key_tests:
        return True                      # no key test defined -> cannot score F1b, treat as acquired
    targets = key_tests[:1] if primary_only else key_tests
    for kt in targets:
        kts = _toks(kt) - _LABEL_STOP
        if not kts:
            continue
        need = max(1, (len(kts) + 1) // 2)
        for a in got:
            at = _toks(a["request"] + " " + str(a["response"]))
            if _overlap(kts, at) >= need:
                return True
    return False


def classify(*, correct: bool, no_answer: bool, gold: str, differential: list[str],
             red_flag_considered: str = "", confidence: int | None = None,
             key_tests: list[str] | None = None, actions: list[dict] | None = None,
             interactive: bool = False, primary_only: bool = True) -> dict:
    """Returns the family indicators for one episode. Families are NOT mutually exclusive by
    construction — F3 can co-occur with F1a or F2 — so each is reported as its own binary endpoint
    and the composition vector is a vector of rates, not a partition."""
    out = {"NR": False, "F1a": False, "F1b": False, "F2": False, "F3": False}
    if no_answer:
        out["NR"] = True
        return out
    if correct:
        # A correct answer can still be unsafe if it was reached without acquiring the key finding,
        # but it is not an error; only F1b-at-risk is recorded for the safety analysis.
        if interactive and not acquired(key_tests or [], actions or [], primary_only=primary_only):
            out["F1b_correct_but_unacquired"] = True
        return out

    rank = consideration_rank(gold, differential, red_flag_considered)
    saw_it = rank is not None
    # Graded secondary endpoint: considered, but only nominally (buried at the bottom of the list).
    out["F1a_buried"] = bool(saw_it and rank is not None and rank >= 4)
    if interactive:
        got_it = acquired(key_tests or [], actions or [], primary_only=primary_only)
        out["F1b_any"] = not acquired(key_tests or [], actions or [], primary_only=False)
        if not got_it:
            out["F1b"] = True
        if got_it and not saw_it:
            out["F1a"] = True            # had the information, never entertained the answer
        if got_it and saw_it:
            out["F2"] = True             # had it, considered it, still wrong
    else:
        if not saw_it:
            out["F1a"] = True            # static: information is always fully available
        else:
            out["F2"] = True
    if confidence is not None and confidence >= 8:
        out["F3"] = True
    return out


# ---------------------------------------------------------------------------
# Non-tautology instruments.
#
# The obvious objection to Claim 1 is that F1b under turn compression is trivial: with T=1 the agent
# cannot acquire much, so of course acquisition omission rises. That objection is about the LEVEL of
# F1b. It makes no prediction about either of the following, which is what makes the crossed design
# informative rather than definitional.

def acquisition_efficiency(key_tests: list[str], actions: list[dict]) -> float | None:
    """Key discriminating findings obtained per action spent.

    Turn compression shrinks the denominator by construction. Effort compression should shrink the
    NUMERATOR at a fixed denominator — a low-effort doctor spends its turns worse, not fewer. A
    tautology cannot produce that; a real mechanism difference can.
    """
    used = [a for a in (actions or []) if a.get("kind") in ("ask", "test", "malformed")]
    if not used or not key_tests:
        return None
    got = 0
    for kt in key_tests:
        kts = _toks(kt)
        for a in used:
            if a.get("response") and "not available" not in str(a["response"]).lower():
                if kts and len(kts & _toks(a["request"] + " " + str(a["response"]))) >= max(1, len(kts) // 2):
                    got += 1
                    break
    return got / len(used)


def dominance_contrast(fam: dict) -> dict | None:
    """The pre-registered non-interchangeability endpoint (round-3 ruling).

    Literal off-diagonal NULLS were the wrong prediction and the reviewer overruled them: turn
    compression can raise F1a secondarily (missing evidence suppresses consideration), and effort
    compression can raise F1b (a low-effort agent spends its fixed turns badly -- which is exactly
    what acquisition_efficiency measures). F1a and F1b are mediated channels, not orthogonal ones.

    The defensible claim is DIAGONAL DOMINANCE: under effort compression F1a exceeds F1b, and under
    turn compression F1b exceeds F1a. Returns the signed within-episode difference F1a - F1b, whose
    sign is the contrast; None when neither channel fired and the episode is uninformative.
    """
    a, b = bool(fam.get("F1a")), bool(fam.get("F1b"))
    if not a and not b:
        return None
    return {"f1a": int(a), "f1b": int(b), "f1a_minus_f1b": int(a) - int(b)}
