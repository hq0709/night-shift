"""Unified LLM client: OpenAI + Anthropic, with disk cache, retry, and cost ledger."""
from __future__ import annotations
import os, json, time, hashlib, threading, pathlib, random
from dataclasses import dataclass, field
from typing import Any

_ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE_DIR = _ROOT / "cache"
LOG_DIR = _ROOT / "logs"
CACHE_DIR.mkdir(exist_ok=True); LOG_DIR.mkdir(exist_ok=True)

def _load_env():
    p = _ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
_load_env()

# USD per 1M tokens: (input, output). Anthropic from claude-api skill (cached 2026-06-24).
PRICING: dict[str, tuple[float, float]] = {
    # --- Anthropic ---
    "claude-opus-5":       (5.00, 25.00),
    "claude-sonnet-5":     (2.00, 10.00),
    "claude-haiku-4-5":    (1.00,  5.00),
    "claude-opus-4-8":     (5.00, 25.00),
    # --- OpenAI (list prices; verify before final budget report) ---
    "gpt-4o-mini":         (0.15,  0.60),
    "gpt-4o":              (2.50, 10.00),
    "gpt-5-nano":          (0.05,  0.40),
    "gpt-5-mini":          (0.25,  2.00),
    "gpt-5":               (1.25, 10.00),
    "gpt-5.4-nano":        (0.05,  0.40),
    "gpt-5.4-mini":        (0.25,  2.00),
    "gpt-5.4":             (1.25, 10.00),
    "gpt-5.5":             (1.25, 10.00),
}
# Anthropic models where `temperature` is rejected with 400 (Claude 5 family / 4.6+).
NO_TEMPERATURE = {"claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-fable-5"}
# OpenAI reasoning models that only accept temperature=1 (default).
OPENAI_NO_TEMP_PREFIXES = ("gpt-5", "o3", "o4")


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_hits: int = 0
    cost_usd: float = 0.0


class Ledger:
    def __init__(self):
        self._lock = threading.Lock()
        self.by_model: dict[str, Usage] = {}

    def add(self, model: str, inp: int, out: int, cached: bool):
        with self._lock:
            u = self.by_model.setdefault(model, Usage())
            u.calls += 1
            if cached:
                u.cached_hits += 1
                return
            u.input_tokens += inp; u.output_tokens += out
            pi, po = PRICING.get(model, (0.0, 0.0))
            delta = inp / 1e6 * pi + out / 1e6 * po
            u.cost_usd += delta
        record_global_spend(delta)

    def total_cost(self) -> float:
        return sum(u.cost_usd for u in self.by_model.values())

    def report(self) -> str:
        lines = [f"{'model':22s} {'calls':>7s} {'cached':>7s} {'in_tok':>10s} {'out_tok':>10s} {'USD':>9s}"]
        for m, u in sorted(self.by_model.items()):
            lines.append(f"{m:22s} {u.calls:7d} {u.cached_hits:7d} {u.input_tokens:10d} {u.output_tokens:10d} {u.cost_usd:9.4f}")
        lines.append(f"{'TOTAL':22s} {'':7s} {'':7s} {'':10s} {'':10s} {self.total_cost():9.4f}")
        return "\n".join(lines)


LEDGER = Ledger()
DAILY_CAP_USD = float(os.environ.get("DAILY_CAP_USD", "100"))

# ---------------------------------------------------------------------------
# Cross-process daily spend.
#
# The in-process LEDGER is per-run, so a $100 "daily cap" enforced against it is not a daily cap at
# all: with four concurrent runs the real day total can reach 4x the cap without anything tripping.
# Discovered 2026-08-28 with a projected day total of ~$124 against a $100 instruction. Spend is
# therefore also journalled to a shared file that every process reads before it starts and appends
# to as it goes.
_SPEND_FILE = _ROOT / "results" / "daily_spend.json"
_spend_lock = threading.Lock()


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def global_spend_usd() -> float:
    try:
        d = json.loads(_SPEND_FILE.read_text())
        return float(d.get(_today(), 0.0))
    except Exception:
        return 0.0


def record_global_spend(amount: float) -> float:
    """Append `amount` to today's shared total and return the new total."""
    if amount <= 0:
        return global_spend_usd()
    with _spend_lock:
        try:
            d = json.loads(_SPEND_FILE.read_text())
        except Exception:
            d = {}
        d[_today()] = round(float(d.get(_today(), 0.0)) + amount, 6)
        _SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _SPEND_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(_SPEND_FILE)
        return d[_today()]


class BudgetExceeded(RuntimeError):
    pass


_clients: dict[str, Any] = {}
_client_lock = threading.Lock()


def _openai():
    with _client_lock:
        if "openai" not in _clients:
            from openai import OpenAI
            _clients["openai"] = OpenAI(timeout=180.0, max_retries=0)
        return _clients["openai"]


def _anthropic():
    with _client_lock:
        if "anthropic" not in _clients:
            import anthropic
            _clients["anthropic"] = anthropic.Anthropic(timeout=180.0, max_retries=0)
        return _clients["anthropic"]


def _cache_key(model, system, messages, temperature, max_tokens, seed, extra) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "messages": messages, "t": temperature,
         "mt": max_tokens, "seed": seed, "x": extra},
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(key: str) -> pathlib.Path:
    d = CACHE_DIR / key[:2]
    d.mkdir(exist_ok=True)
    return d / f"{key}.json"


def chat(model: str, messages: list[dict], system: str | None = None,
         temperature: float | None = None, max_tokens: int = 2048,
         seed: int | None = None, use_cache: bool = True,
         max_retries: int = 5, tag: str = "") -> dict:
    """Returns {'text', 'input_tokens', 'output_tokens', 'model', 'cached'}."""
    key = _cache_key(model, system, messages, temperature, max_tokens, seed, tag)
    cp = _cache_path(key)
    if use_cache and cp.exists():
        try:
            r = json.loads(cp.read_text())
            LEDGER.add(model, r["input_tokens"], r["output_tokens"], cached=True)
            r["cached"] = True
            return r
        except Exception:
            pass

    g = global_spend_usd()
    if g > DAILY_CAP_USD:
        raise BudgetExceeded(
            f"cross-process day spend ${g:.2f} exceeds cap ${DAILY_CAP_USD} "
            f"(this process: ${LEDGER.total_cost():.2f})")

    last_err = None
    for attempt in range(max_retries):
        try:
            if model.startswith("claude"):
                kw: dict[str, Any] = dict(model=model, max_tokens=max_tokens,
                                          messages=messages)
                if system:
                    kw["system"] = system
                if temperature is not None and model not in NO_TEMPERATURE:
                    kw["temperature"] = temperature
                resp = _anthropic().messages.create(**kw)
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                inp, out = resp.usage.input_tokens, resp.usage.output_tokens
            else:
                msgs = ([{"role": "system", "content": system}] if system else []) + messages
                kw = dict(model=model, messages=msgs)
                if model.startswith(OPENAI_NO_TEMP_PREFIXES):
                    kw["max_completion_tokens"] = max_tokens
                else:
                    kw["max_tokens"] = max_tokens
                    if temperature is not None:
                        kw["temperature"] = temperature
                    if seed is not None:
                        kw["seed"] = seed
                resp = _openai().chat.completions.create(**kw)
                text = resp.choices[0].message.content or ""
                inp, out = resp.usage.prompt_tokens, resp.usage.completion_tokens

            r = {"text": text, "input_tokens": inp, "output_tokens": out,
                 "model": model, "cached": False}
            if use_cache:
                cp.write_text(json.dumps(r, ensure_ascii=False))
            LEDGER.add(model, inp, out, cached=False)
            return r
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            if any(s in msg for s in ("invalid_request", "does not exist", "not supported",
                                      "Unsupported", "invalid_api_key")):
                raise
            time.sleep(min(2 ** attempt + random.random(), 30))
    raise RuntimeError(f"chat failed after {max_retries} retries: {last_err}")
