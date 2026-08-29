"""The measurement contract.

Round-1 review, confound #2: low effort produces terser free text, which would mechanically deflate
differential breadth and inflate red-flag misses — a verbosity artifact masquerading as an omission
effect. The fix is a FIXED structured-output schema, identical at every effort level, so the
elicitation of every measured field is constant across conditions and only the *content* varies.

K_MAX is the cap on the differential list, not a required length: forcing exactly K would destroy
breadth as a measure. The field is always requested; how many entries the model puts in it is the
dependent variable.
"""
from __future__ import annotations

K_MAX = 5

EPISODE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "differential": {
            "type": "array",
            "description": (f"Every diagnosis or option you actually considered, most likely first, "
                            f"at most {K_MAX}. List only what you genuinely weighed."),
            "items": {"type": "string"},
        },
        "red_flag_considered": {
            "type": "string",
            "description": ("The single most dangerous can't-miss condition you actively ruled out for "
                            "this presentation, or the exact string 'none' if you ruled none out."),
        },
        "key_findings_used": {
            "type": "array",
            "description": "The findings from the vignette that drove your decision.",
            "items": {"type": "string"},
        },
        "final_answer": {
            "type": "string",
            "description": "The single option letter you commit to. One capital letter only.",
        },
        "confidence": {
            "type": "integer",
            "description": "Your confidence that final_answer is correct, 1 (guess) to 10 (certain).",
        },
    },
    "required": ["differential", "red_flag_considered", "key_findings_used",
                 "final_answer", "confidence"],
}

SCHEMA_NAME = "clinical_episode"


def episode_schema(answer_kind: str = "letter") -> dict:
    """`letter` for MCQ testbeds, `diagnosis` for the interactive OSCE testbed. Everything except
    the final_answer description is identical, so the two testbeds share one measurement contract."""
    import copy
    sch = copy.deepcopy(EPISODE_SCHEMA)
    if answer_kind == "diagnosis":
        sch["properties"]["final_answer"]["description"] = (
            "The single most likely diagnosis you commit to. A diagnosis name only.")
    return sch

# Identical instruction text at every effort level. Nothing here mentions length, brevity or budget:
# the budget manipulation must never leak into the measurement contract.
CONTRACT_NOTE = (
    "Return your decision as JSON matching the required schema. Fill every field on every question, "
    "however much or little you reasoned."
)


def openai_response_format(answer_kind: str = "letter") -> dict:
    return {"type": "json_schema",
            "json_schema": {"name": SCHEMA_NAME, "strict": True,
                            "schema": episode_schema(answer_kind)}}


def anthropic_output_config(answer_kind: str = "letter") -> dict:
    # Verified 2026-08-28: `name` inside output_config.format is rejected with
    # "output_config.format.name: Extra inputs are not permitted". Shape is type + schema only.
    return {"format": {"type": "json_schema", "schema": episode_schema(answer_kind)}}


def parse(text: str, options: dict | None = None) -> dict:
    """Read-out from a schema response. Falls back to the regex extractor on parse failure so a
    non-response is recorded as NON-RESPONSE rather than silently dropped."""
    import json, re
    valid = set(options) if options else None
    raw = (text or "").strip()
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        from envs.qa import extract
        e = extract(raw, options or {})
        return {**e, "parsed": False, "red_flag_considered": "", "key_findings": [],
                "n_findings": 0}

    fa = str(obj.get("final_answer", "")).strip()
    if valid is None:                     # free-text diagnosis testbed
        letter = fa or None
    else:
        letter = None
        for ch in fa.upper():
            if ch in valid:
                letter = ch
                break
    diff = [str(x).strip() for x in (obj.get("differential") or []) if str(x).strip()]
    kf = [str(x).strip() for x in (obj.get("key_findings_used") or []) if str(x).strip()]
    try:
        conf = int(obj.get("confidence"))
        conf = conf if 1 <= conf <= 10 else None
    except Exception:
        conf = None
    return {"letter": letter, "confidence": conf, "differential": diff, "breadth": len(diff),
            "no_answer": letter is None, "parsed": True,
            "red_flag_considered": str(obj.get("red_flag_considered", "")).strip(),
            "key_findings": kf, "n_findings": len(kf)}
