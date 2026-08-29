# Elicited Confidence Is Not a Safety Signal

Code and paper for a study of verbalized-confidence reliability in clinical language models.

**Finding.** The discriminative power of elicited confidence is task-dependent, not a fixed model
property. On standard clinical items it separates correct from incorrect answers (AUROC 0.67–0.70);
at the capability frontier it collapses to chance (0.48–0.54); on cases carrying can't-miss
diagnoses it degrades *beyond* what difficulty predicts (ΔAUROC −0.082 / −0.063 vs accuracy-matched
controls) and is nearly uninformative about whether the can't-miss diagnosis was missed at all
(0.532 / 0.567). In interactive consultation, where the model gathers its own evidence,
discrimination rises to 0.75/0.80. Confidence-gated deferral therefore fails on precisely the cases
it exists to catch.

Approximately 30,000 episodes · 5 models · 2 vendors · 9 evaluation sets · static and interactive
settings · three ablations of the elicitation schema.

**A correction recorded here because it changed a headline claim.** An earlier version of this
analysis reported a missed-red-flag rate of 40--47% and near-chance discrimination on that endpoint.
Those numbers were computed on incomplete records: the elicited differential was not being persisted,
so the endpoint of Eq. (4), defined over `differential ∪ {rho}`, was in fact evaluated on `rho`
alone. With the full record the hit rate is 92% and discrimination is 0.667--0.687. The manuscript
reports the corrected values.

## Layout

```
main.tex   manuscript (root level, for Overleaf GitHub sync)
figs/      figures
src/
  common/llm.py            unified OpenAI+Anthropic client: disk cache, retry, cost ledger
  budgets/dial.py          effort/budget control across provider APIs; 429-aware backoff
  envs/schema.py           the fixed structured-output measurement contract
  envs/qa.py               MedAgentsBench loading, stratified sampling, read-out
  envs/clinic.py           interactive OSCE consultation (turn budget, cross-vendor patient agent)
  envs/medcalc.py          procedural task family; slip vs knowledge-error split
  envs/general.py          independent corpora (MMLU-Pro, ARC, GSM8K) + RedFlag + AgentClinic
  taxonomy/sentinels.py    48 stem sentinel patterns for can't-miss presentations
  taxonomy/redflag_list.py 69 can't-miss conditions with aliases
  taxonomy/families.py     error-family classification, consideration/acquisition split
  experiments/             runners and analyses
  scripts/                 RedFlag construction pipeline, bibliography tooling
data/      RedFlag probe (99 items, 44 conditions), annotations, verification worksheet
```

## Reproducing

Set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in a `.env` at the repo root (never committed).
Benchmarks are fetched from their public sources; MedAgentsBench and AgentClinic are expected under
`external/`.

```bash
python3 src/experiments/run_axisA.py --task mab --n 300 --models gpt-5.4-mini --seeds 3
python3 src/experiments/run_interactive_conf.py --models gpt-5.4-mini --n 214 --seeds 3
python3 src/experiments/analyze_itt.py --path results/axisA_gpt54mini.jsonl
python3 src/experiments/make_paper_figs.py
```

Every API response is cached to disk, so reruns are free for work already done.

## RedFlag-99

Filtering benchmark items by gold *diagnosis* recovers few high-acuity cases, because exam gold
answers are usually management steps: matching a can't-miss list against gold labels returned 41
items from 7,956. The probe is therefore built from **sentinel features in the stem** — a mechanical
criterion requiring no clinical judgement at filter time — then confirmed, canonicalised to a closed
vocabulary, and audited. Three independent automated annotators give Fleiss κ = 0.487 with 77/99
unanimous; all principal analyses are repeated on the unanimous subset.

**No physician reviewed these items.** `data/redflag_verification_sheet.tsv` is prepared for that
review.

## Building the paper

```bash
tectonic -X compile main.tex
```
