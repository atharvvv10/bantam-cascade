# Bantam Cascade

*Bantam Cascade finds the cheapest small-CPU-model + big-LLM fallback cascade that provably meets a user's accuracy requirement, or says honestly that it cannot certify one.*

## Status: Phase 2 — CPU students, cascade, and provider choice

This repository is being built phase by phase. Phase 0 (scaffold, config, splits, ledger, CLI), Phase 1 (Groq and Gemini Decisions teacher clients, cache, prompt/parser, dry-run estimator, selection), and Phase 2 (CPU-only students, calibration, cascade measurement) are complete. Statistical certification arrives in Phase 3. Every number that appears here comes from a results file produced by a real run; numbers not yet measured are marked **TBD**.

## What the finished tool does

1. Takes a labeled text-classification dataset plus a target accuracy, confidence level, and p95 latency budget.
2. Trains small CPU-only students, calibrates them, and builds a cascade (student answers when confident, teacher otherwise).
3. Uses finite-sample statistics against **gold human labels** to certify which configurations meet the target.
4. Returns the cheapest certified configuration, a Pareto frontier, or an honest `INSUFFICIENT_EVIDENCE` / `NO_FEASIBLE_CONFIG` status.

All cost figures are **ESTIMATES** derived only from prices in the user's config. The word "certified" always appears with its assumptions attached.

## Quickstart (TBD)

```text
pip install -e ".[dev]"
bantam-cascade --help
bantam-cascade init
```

Full pipeline usage will be documented after the relevant phases are complete.

## Teacher labeling (Phase 1)

Copy `.env.example` to `.env` in the project root and put your provider key in it. Use `GROQ_API_KEY` for Groq or `GEMINI_API_KEY` for Gemini Decisions. The file is git-ignored; never commit or print keys. The **config**, not the key, pins the provider and model.

```text
bantam-cascade label --config configs/banking77.yaml --dry-run
bantam-cascade label --config configs/banking77.yaml --limit 200
bantam-cascade select-teacher --config configs/banking77.yaml --limit 200

bantam-cascade label --config configs/banking77-gemini.yaml --dry-run
bantam-cascade label --config configs/banking77-gemini.yaml --limit 200
```

The shipped Groq config pins `openai/gpt-oss-20b`; the Gemini config pins `gemini-decision-1`, which answers a fixed 2–255-way `choice` question with calibrated probabilities and **zero output tokens**. Both use user-supplied prices (currently `0`) and user-supplied rate limits. Dry run makes no API calls and needs no key.

Measured on the same 200 gold calibration labels (local run artifacts are git-ignored):

| Teacher | Accuracy | Invalid | Input tokens | Output tokens | Mean latency |
|---|---:|---:|---:|---:|---:|
| Groq `openai/gpt-oss-20b` | 0.745 | 0 | 125,883 | 8,288 | 490.4 ms |
| Gemini `gemini-decision-1` | 0.765 | 0 | 158,963 | 0 | 851.2 ms |

Sources: `runs/phase1/teacher_labels.jsonl` and `runs/phase1-gemini/teacher_labels.jsonl`. The four-example gap is a measurement, not a statistical certificate.

## Students and cascade (Phase 2)

```text
bantam-cascade train-students --config configs/banking77.yaml --run-dir runs/phase2
bantam-cascade build-cascade --config configs/banking77.yaml --run-dir runs/phase2 --dry-run
bantam-cascade build-cascade --config configs/banking77.yaml --run-dir runs/phase2 --calibration-labels runs/phase1/teacher_labels.jsonl
```

Students are `static_embed` (TF-IDF + logistic regression, no downloads) and `minilm_onnx_int8` (MiniLM ONNX quantized for CPU). MiniLM needs the optional ONNX dependencies: `pip install -e ".[students]"`. Students train on **gold human labels**; the teacher is used only as the cascade fallback and is also scored against gold. `build-cascade` reads the 200 calibration labels from Phase 1 and labels the certification split with the teacher (resumable via the cache). All outputs go to `runs/phase2/`. These are measurements, not statistical certificates.

With the Gemini fallback and its matching calibration labels on 300 certification examples (`runs/phase2-gemini/cascade_results.jsonl`): `static_embed` measured 0.837 accuracy with 11 fallback calls and 1.5 ms p95; `minilm_onnx_int8` measured 0.920 accuracy with 1 fallback call and 7.3 ms p95. Both are measurements, not certificates.

## How certification works

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the plain-English explanation and the exact statistical assumptions.

## Limitations

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md). Nothing here is a guarantee about deployment data that differs from the certified data.

## Prior work and credits

Bantam Cascade does **not** claim novelty for distillation, cascades, or threshold selection with guarantees. It builds on TinyDistill, bounded-distill, FrugalGPT, and Learn-then-Test (Angelopoulos, Bates, Candès, Jordan, Lei). Full citations are in [CITATION.cff](CITATION.cff) and [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Naming

The name [`bantam`](https://pypi.org/project/bantam/) on PyPI belongs to an unrelated microservices package. That's why this project uses `bantam-cascade` for its distribution, CLI, and repository, and `bantam_cascade` for its Python import.

## License

Apache-2.0. See [LICENSE](LICENSE).
