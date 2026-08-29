"""The fatigue dial: one call interface over OpenAI + Anthropic reasoning-budget controls.

Verified 2026-08-28 (scripts/probe_budget_dials.py):
  * OpenAI gpt-5.x : reasoning_effort in {none,low,medium,high,xhigh}; 'minimal' -> 400.
                     usage.completion_tokens_details.reasoning_tokens is exact.
  * Anthropic C5   : thinking.type.enabled -> 400. Only thinking={"type":"adaptive"} +
                     output_config.effort in {low,medium,high,xhigh,max}. Raw thinking text
                     is NOT returned (display defaults to omitted), so reasoning tokens are
                     DERIVED as output_tokens - answer_tokens.
  * Anthropic <=4.5: thinking.budget_tokens works, minimum 1024.
"""
from __future__ import annotations
import os, re, sys, json, time, hashlib, pathlib, random, threading
from dataclasses import dataclass, asdict
from typing import Any, Literal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import llm  # noqa: E402  (loads .env, LEDGER, cache dir)

# A single episode may wait out many rate-limit windows; only a sustained inability to schedule
# counts as a failure. 40 waits x up to 90s is roughly an hour of patience per episode.
MAX_RATE_LIMIT_WAITS = 40

class ProviderUnavailable(RuntimeError):
    """The account cannot serve any request (billing/quota). Distinct from an item-level error:
    every further call to that provider will fail, so a run should stop rather than grind on."""


_PROVIDER_DOWN: set[str] = set()


def provider_down(model: str) -> bool:
    return ("anthropic" if model.startswith("claude") else "openai") in _PROVIDER_DOWN


# Per-model output ceilings. A non-reasoning model rejects the large max_tokens the reasoning
# models need: gpt-4o-mini caps at 16,384 and returned 400 "max_tokens is too large" for every
# call in the first Stage-1 sweep, silently voiding that whole arm.
MAX_OUT = {"gpt-4o-mini": 16000, "gpt-4o": 16000}

OPENAI_EFFORTS = ["none", "low", "medium", "high", "xhigh"]
ANTHROPIC_C5_EFFORTS = ["low", "medium", "high", "xhigh", "max"]

# Which family a model id belongs to.
# Models on the adaptive-thinking / output_config.effort API (budget_tokens rejected).
_C5 = {"claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-mythos-5",
       "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6"}

def family(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic_c5" if model in _C5 else "anthropic_legacy"
    return "openai_reasoning" if model.startswith(("gpt-5", "o1", "o3", "o4")) else "openai_chat"


@dataclass
class Episode:
    """One doctor decision under one budget setting. Everything the analysis needs."""
    model: str
    axis: str                 # 'A_effort' | 'A_instructed' | 'A_hardcap' | 'B_turns' | 'C_shift'
    setting: str              # the instructed level, verbatim
    seed: int
    text: str                 # visible answer
    reasoning_tokens: int     # measured; exact (OpenAI) or derived (Anthropic)
    reasoning_exact: bool     # whether the count is provider-reported or derived
    output_tokens: int
    input_tokens: int
    finish: str
    truncated: bool           # answer censored by a hard cap
    latency_s: float
    cached: bool
    item_id: str = ""
    extra: dict | None = None


# ---------------------------------------------------------------- cache
def _key(**kw) -> str:
    return hashlib.sha256(json.dumps(kw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def _cpath(k: str) -> pathlib.Path:
    d = llm.CACHE_DIR / "dial" / k[:2]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{k}.json"


_tok_cache: dict[str, int] = {}
_tok_lock = threading.Lock()

_OVERHEAD: dict[str, int] = {}

def _msg_overhead(model: str) -> int:
    """count_tokens wraps text in a user message; that envelope must not be charged to thinking."""
    if model not in _OVERHEAD:
        try:
            a = llm._anthropic().messages.count_tokens(
                model=model, messages=[{"role": "user", "content": "a"}]).input_tokens
            b = llm._anthropic().messages.count_tokens(
                model=model, messages=[{"role": "user", "content": "a a a a a a a a a a"}]).input_tokens
            # b - a == 9 real tokens; overhead = a - 1
            _OVERHEAD[model] = max(0, a - 1) if b > a else 0
        except Exception:
            _OVERHEAD[model] = 0
    return _OVERHEAD[model]


def _anthropic_answer_tokens(model: str, text: str) -> int:
    """Token count of the visible answer, so derived_thinking = output_tokens - this."""
    if not text:
        return 0
    k = hashlib.sha256((model + "\x00" + text).encode()).hexdigest()
    with _tok_lock:
        if k in _tok_cache:
            return _tok_cache[k]
    try:
        n = llm._anthropic().messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text}]).input_tokens
        n = max(0, n - _msg_overhead(model))
    except Exception:
        n = max(1, len(text) // 4)
    with _tok_lock:
        _tok_cache[k] = n
    return n


# ---------------------------------------------------------------- instructed-token protocol
INSTRUCTED_TMPL = (
    "You are working under a strict reasoning budget. Think through this in AT MOST {n} tokens "
    "of reasoning, then give your final answer. Do not exceed the budget."
)
INSTRUCTED_ZERO = (
    "Answer immediately with no reasoning, no explanation and no working. Give only the final answer."
)


def ask(model: str, prompt: str, *, axis: str, setting: str, seed: int = 0,
        system: str | None = None, max_tokens: int = 16000, item_id: str = "",
        use_cache: bool = True, max_retries: int = 7,
        structured: bool | str = False, effort: str | None = None,
        _escalated: bool = False) -> Episode:
    """One budgeted call. `setting` semantics depend on `axis`:
         A_effort     -> an effort level name valid for the model family
         A_instructed -> an integer-as-string token budget (0 means 'answer immediately')
         A_hardcap    -> an integer-as-string cap on max output tokens
    `structured` is False, or "letter" (MCQ) / "diagnosis" (interactive OSCE) to enforce the
    fixed measurement schema.
    """
    kind = structured if isinstance(structured, str) else "letter"
    # Round-3 review: "logging plus a high ceiling is not enough by itself. Add an automatic
    # rerun/escalation rule." A truncated episode is a censored measurement, not a datum, so the
    # call is retried once at a doubled ceiling before the truncation is accepted as real.
    # Claim 1 crosses effort with the turn budget, so a non-A axis still needs the effort knob set.
    # Without this the B-axis runs silently at the provider default and the crossed design collapses.
    # The cache key MUST include the active output schema. Different confidence-elicitation
    # formats send an identical prompt and differ only in the JSON schema, so without this they
    # collide on one cache entry and every format silently returns the first one's answer.
    # Observed 2026-08-28: B_probability read back A_integer's "9" and C_verbal read back the same
    # integer and parsed to NaN -- which would have produced a fake "all formats behave identically"
    # result, the exact conclusion the experiment exists to test.
    _schema_fp = ""
    if structured:
        from envs import schema as _sch
        _schema_fp = hashlib.sha256(
            json.dumps(_sch.episode_schema(kind), sort_keys=True).encode()).hexdigest()[:16]
    ck = _key(m=model, p=prompt, s=system, ax=axis, st=setting, sd=seed, mt=max_tokens,
              sc=structured, ef=effort, sfp=_schema_fp)
    cp = _cpath(ck)
    if use_cache and cp.exists():
        try:
            d = json.loads(cp.read_text()); d["cached"] = True
            llm.LEDGER.add(model, d["input_tokens"], d["output_tokens"], cached=True)
            return Episode(**d)
        except Exception:
            pass

    # This is the hot path for every experiment; `llm.chat()` is not. Checking the in-process
    # LEDGER here made the "daily cap" vacuous for the runs that actually spend the money -- with
    # several concurrent runs each stayed far under the cap while the day total passed it.
    # Observed 2026-08-28: day reached $97 with two runs reporting $2.14 and $5.96 of their own.
    _g = llm.global_spend_usd()
    if _g > llm.DAILY_CAP_USD:
        raise llm.BudgetExceeded(
            f"cross-process day spend ${_g:.2f} exceeds cap ${llm.DAILY_CAP_USD} "
            f"(this process: ${llm.LEDGER.total_cost():.2f})")

    if provider_down(model):
        raise ProviderUnavailable(
            f"{'Anthropic' if model.startswith('claude') else 'OpenAI'} already marked unusable "
            f"this process; refusing to spend further calls against it")

    fam = family(model)
    max_tokens = min(max_tokens, MAX_OUT.get(model, max_tokens))
    user = prompt
    if axis == "A_instructed":
        n = int(setting)
        user = (INSTRUCTED_ZERO if n == 0 else INSTRUCTED_TMPL.format(n=n)) + "\n\n" + prompt

    # Rate-limit retries must NOT consume the error-retry budget.
    #
    # Measured 2026-08-28: with a shared 7-retry budget, a MedCalc smoke run lost 4 of 56 episodes
    # to 429 -- and the losses concentrated in the `xhigh` cells, because those calls are the
    # largest and so the most likely to be throttled. That is non-random attrition in precisely the
    # condition the study cares most about, and it would bias every per-operating-point rate. 429 is
    # a scheduling signal, not a failure, so it gets its own generous budget.
    last = None
    rate_limit_waits = 0
    attempt = 0
    while attempt < max_retries:
        t0 = time.time()
        try:
            if fam.startswith("openai"):
                kw: dict[str, Any] = dict(model=model,
                                          messages=([{"role": "system", "content": system}] if system else [])
                                                   + [{"role": "user", "content": user}])
                cap = max_tokens
                if fam == "openai_reasoning":
                    if axis == "A_effort":
                        if setting != "none":
                            kw["reasoning_effort"] = setting
                        else:
                            kw["reasoning_effort"] = "none"
                    elif axis == "A_instructed":
                        kw["reasoning_effort"] = "high"   # let the prompt do the limiting
                    elif axis == "A_hardcap":
                        kw["reasoning_effort"] = "high"; cap = int(setting)
                    elif effort:
                        kw["reasoning_effort"] = effort
                    kw["max_completion_tokens"] = cap
                    if structured:
                        from envs import schema as _sch
                        kw["response_format"] = _sch.openai_response_format(kind)
                else:
                    if axis == "A_hardcap":
                        cap = int(setting)
                    kw["max_tokens"] = cap
                    if structured:
                        from envs import schema as _sch
                        kw["response_format"] = _sch.openai_response_format(kind)
                    kw["temperature"] = 0.3
                    if seed is not None:
                        kw["seed"] = seed
                r = llm._openai().chat.completions.create(**kw)
                ch = r.choices[0]
                text = ch.message.content or ""
                det = getattr(r.usage, "completion_tokens_details", None)
                rt = int(getattr(det, "reasoning_tokens", 0) or 0)
                ep = Episode(model=model, axis=axis, setting=setting, seed=seed, text=text,
                             reasoning_tokens=rt, reasoning_exact=(fam == "openai_reasoning"),
                             output_tokens=r.usage.completion_tokens,
                             input_tokens=r.usage.prompt_tokens, finish=ch.finish_reason,
                             truncated=(ch.finish_reason == "length"),
                             latency_s=round(time.time() - t0, 2), cached=False, item_id=item_id)
            else:
                kw = dict(model=model, max_tokens=max_tokens,
                          messages=[{"role": "user", "content": user}])
                if system:
                    kw["system"] = system
                if fam == "anthropic_c5":
                    if axis == "A_effort":
                        if setting == "off":
                            kw["thinking"] = {"type": "disabled"}
                        else:
                            kw["thinking"] = {"type": "adaptive", "display": "summarized"}
                            oc_: dict[str, Any] = {"effort": setting}
                            if structured:
                                from envs import schema as _sch
                                oc_.update(_sch.anthropic_output_config(kind))
                            kw["extra_body"] = {"output_config": oc_}
                    elif axis == "A_instructed":
                        kw["thinking"] = {"type": "disabled"}   # prompt is the only dial available
                    elif axis == "A_hardcap":
                        kw["thinking"] = {"type": "adaptive"}; kw["max_tokens"] = int(setting)
                    elif effort:
                        kw["thinking"] = {"type": "adaptive", "display": "summarized"}
                        oc2: dict[str, Any] = {"effort": effort}
                        if structured:
                            from envs import schema as _sch
                            oc2.update(_sch.anthropic_output_config(kind))
                        kw["extra_body"] = {"output_config": oc2}
                    if structured and "extra_body" not in kw:
                        from envs import schema as _sch
                        kw["extra_body"] = {"output_config": _sch.anthropic_output_config(kind)}
                else:  # pre-4.6: real budget_tokens, minimum 1024
                    if axis == "A_effort":
                        bt = int(setting)
                        if bt <= 0:
                            kw["thinking"] = {"type": "disabled"}
                        else:
                            kw["thinking"] = {"type": "enabled", "budget_tokens": max(1024, bt)}
                            kw["max_tokens"] = max(1024, bt) + 2000
                    elif axis == "A_hardcap":
                        kw["max_tokens"] = int(setting)
                r = llm._anthropic().messages.create(**kw)
                text = "".join(b.text for b in r.content if b.type == "text")
                raw_think = "".join(getattr(b, "thinking", "") or "" for b in r.content if b.type == "thinking")
                if raw_think:
                    rt = _anthropic_answer_tokens(model, raw_think); exact = True
                else:
                    rt = max(0, r.usage.output_tokens - _anthropic_answer_tokens(model, text)); exact = False
                ep = Episode(model=model, axis=axis, setting=setting, seed=seed, text=text,
                             reasoning_tokens=rt, reasoning_exact=exact,
                             output_tokens=r.usage.output_tokens, input_tokens=r.usage.input_tokens,
                             finish=r.stop_reason or "", truncated=(r.stop_reason == "max_tokens"),
                             latency_s=round(time.time() - t0, 2), cached=False, item_id=item_id)

            if ep.truncated and not _escalated and axis != "A_hardcap":
                # A_hardcap is *meant* to censor; everywhere else truncation is an artifact.
                return ask(model, prompt, axis=axis, setting=setting, seed=seed, system=system,
                           max_tokens=min(max_tokens * 2, 128000), item_id=item_id,
                           use_cache=use_cache, max_retries=max_retries, structured=structured,
                           effort=effort, _escalated=True)
            if use_cache:
                cp.write_text(json.dumps(asdict(ep), ensure_ascii=False))
            llm.LEDGER.add(model, ep.input_tokens, ep.output_tokens, cached=False)
            return ep
        except Exception as e:
            last = e
            m = str(e)
            # A billing failure arrives as 400 invalid_request_error, which the generic rule below
            # treats as a bad request and charges to the individual item. It is nothing of the kind:
            # the account is down, every subsequent call to that provider will fail, and a long run
            # will keep spending on the OTHER provider while silently logging thousands of
            # "per-item errors". Observed 2026-08-28 -- 19 such errors on Anthropic mid-sweep, with
            # the true cause only visible by reproducing the call by hand. Fail loudly and globally.
            if any(t in m.lower() for t in ("credit balance is too low", "billing", "quota",
                                            "insufficient_quota", "payment required")):
                _PROVIDER_DOWN.add("anthropic" if model.startswith("claude") else "openai")
                raise ProviderUnavailable(
                    f"{'Anthropic' if model.startswith('claude') else 'OpenAI'} account is "
                    f"unusable (billing/quota), not an item-level error: {m[:200]}")
            if any(s in m for s in ("invalid_request", "does not exist", "not supported",
                                    "Unsupported", "invalid_api_key", "does not support")):
                raise
            # Verified 2026-08-28: gpt-5.4-mini TPM limit is 200k/min on this org, so 429 is the
            # normal steady state at high concurrency, not an error.
            rate_limited = "429" in m or "rate limit" in m.lower()
            if rate_limited:
                rate_limit_waits += 1
                if rate_limit_waits > MAX_RATE_LIMIT_WAITS:
                    raise RuntimeError(
                        f"gave up after {rate_limit_waits} rate-limit waits: {m[:160]}")
                # Honour the server's own Retry-After when it states one.
                mo = re.search(r"try again in ([0-9.]+)s", m) or \
                    re.search(r"retry[- ]after[\"': ]+([0-9.]+)", m, re.I)
                wait = float(mo.group(1)) + 1.0 if mo else 15.0
                time.sleep(min(wait + random.random() * 5, 90))
                continue                     # does not consume the error-retry budget
            attempt += 1
            time.sleep(min(2 ** attempt + random.random(), 30))
    raise RuntimeError(f"ask failed after {max_retries}: {last}")
