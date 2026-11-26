# Bantam Cascade

*Bantam Cascade finds the cheapest small-CPU-model + big-LLM fallback cascade that provably meets a user's accuracy requirement, or says honestly that it cannot certify one.*

## Status: Phases 0–11 — complete

The original plan (Phases 0–3) is complete, and the extended best-version roadmap (Phases 4–11) is complete too. Every number below comes from a real run artifact. The tool now has **one genuine 95% certificate** (SMS Spam) and one honest non-certificate (Banking77). Nothing is claimed that the statistics do not support.

## What the finished tool does

1. Takes a labeled text-classification dataset plus a target accuracy, confidence level, and p95 latency budget.
2. Trains small CPU-only students, calibrates them, and builds a cascade (student answers when confident, teacher otherwise).
3. Uses finite-sample statistics against **gold human labels** to certify which configurations meet the target.
4. Returns the cheapest certified configuration, a Pareto frontier, or an honest `INSUFFICIENT_EVIDENCE` / `NO_FEASIBLE_CONFIG` status.

All cost figures are **ESTIMATES** derived only from prices in the user's config. The word "certified" always appears with its assumptions attached.

## Quickstart

```text
pip install -e ".[dev]"
bantam-cascade --help

# Certified path (SMS Spam, pinned Groq teacher, word_tfidf student)
bantam-cascade init --config configs/sms-spam-groq.yaml --run-dir runs/sms-groq
bantam-cascade label --config configs/sms-spam-groq.yaml --run-dir runs/sms-groq
bantam-cascade train-students --config configs/sms-spam-groq.yaml --run-dir runs/sms-groq
bantam-cascade build-cascade --config configs/sms-spam-groq.yaml --run-dir runs/sms-groq
bantam-cascade certify --config configs/sms-spam-groq.yaml --run-dir runs/sms-groq

# Offline demo of a completed run
bantam-cascade demo "claim your free prize now" --run-dir runs/sms-groq
```

`certify` exits non-zero when it cannot issue a certificate; that is an honest result, not a crash.

## Headline result: certified SMS Spam

A single pre-registered `word_tfidf` cascade was tested once on 300 untouched examples from `ucirvine/sms_spam`:

| Item | Value |
| --- | --- |
| Target / confidence | 95% / 95% |
| Measured accuracy | 0.987 (4 errors / 300) |
| Hoeffding-Bentkus p-value | 0.001878 (`delta = 0.05`, 0 prior looks) |
| Fallback calls | 0 |
| Estimated cost | $0.000000 from configured prices |
| p95 latency | 0.7 ms |
| Status | **`CERTIFIED`** |

The threshold was frozen in `runs/sms-groq/preregistration.json` before the certification split was read. The result is conditional on the assumptions in [docs/METHODOLOGY.md](docs/METHODOLOGY.md); see [docs/RESULTS.md](docs/RESULTS.md) for the full record and the separate Banking77 non-certificate.

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

## Students and cascade (Phases 2 and 4)

```text
bantam-cascade train-students --config configs/banking77.yaml --run-dir runs/phase2
bantam-cascade build-cascade --config configs/banking77.yaml --run-dir runs/phase2 --dry-run
bantam-cascade build-cascade --config configs/banking77.yaml --run-dir runs/phase2 --calibration-labels runs/phase1/teacher_labels.jsonl
```

Students are `static_embed` (word TF-IDF + logistic regression, no downloads), `word_tfidf` (word 1–2 gram TF-IDF + logistic regression, the certified student), and `minilm_onnx_int8`/`fp32` (MiniLM ONNX for CPU). MiniLM needs the optional ONNX dependencies: `pip install -e ".[students]"`. Students train on **gold human labels**; the teacher is used only as the cascade fallback and is also scored against gold. `build-cascade` reads the 200 calibration labels and labels the certification split with the teacher (resumable via the cache). All outputs go to the selected `--run-dir`. Measurements are not certificates unless the pre-registered Learn-then-Test procedure accepts them.

With the Gemini fallback and its matching calibration labels on 300 certification examples (`runs/phase2-gemini/cascade_results.jsonl`): `static_embed` measured 0.837 accuracy with 11 fallback calls and 1.5 ms p95; `minilm_onnx_int8` measured 0.920 accuracy with 1 fallback call and 7.3 ms p95. Both are measurements, not certificates.

## Certification (Phase 3)

```text
bantam-cascade certify --config configs/banking77-gemini.yaml \
  --run-dir runs/phase2-gemini \
  --calibration-labels runs/phase1-gemini/teacher_labels.jsonl
```

`certify` verifies that saved teacher rows carry the dataset's gold labels, recomputes calibration statistics using only the thresholds already learned from the 200-example calibration split, then runs Learn-then-Test fixed-sequence testing on the 300-example certification split. The candidate order comes from calibration data only; no certification data is used to change it. Results are written to `<run-dir>/certification_result.json`, and looks at the split are tracked in `<run-dir>/certification_ledger.sqlite3` (reuse requires `--allow-reuse` and divides δ by the total number of looks).

Real run against the configured 95% accuracy target (`runs/phase2-gemini/certification_result.json`):

| Student | Errors / 300 | Measured accuracy | HB p-value | Status |
|---|---:|---:|---:|---|
| `minilm_onnx_int8` | 24 | 0.920 | 1.000 | not rejected |
| `static_embed` | 49 | 0.837 | 1.000 | not rejected |

Outcome: `INSUFFICIENT_EVIDENCE`. At δ = 0.05, neither cascade can reject the hypothesis that its true error rate exceeds 5%. The tool does **not** claim a certificate here.

## How certification works

Learn-then-Test turns risk control into hypothesis testing. With `α = 1 - minimum_accuracy`, each candidate tests `H: risk > α`. Its p-value uses the Hoeffding-Bentkus bound, and a single-start fixed-sequence procedure controls family-wise error at `δ = 1 - confidence_level` (with a further multiple-looks correction on reuse). See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the exact formula and assumptions.

## Limitations

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md). Nothing here is a guarantee about deployment data that differs from the certified data.

## Prior work and credits

Bantam Cascade does **not** claim novelty for distillation, cascades, or threshold selection with guarantees. It builds on TinyDistill, bounded-distill, FrugalGPT, and Learn-then-Test (Angelopoulos, Bates, Candès, Jordan, Lei). Full citations are in [CITATION.cff](CITATION.cff) and [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Naming

The name [`bantam`](https://pypi.org/project/bantam/) on PyPI belongs to an unrelated microservices package. That's why this project uses `bantam-cascade` for its distribution, CLI, and repository, and `bantam_cascade` for its Python import.

## License

Apache-2.0. See [LICENSE](LICENSE).
