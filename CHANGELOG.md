# Changelog

All notable changes to Bantam Cascade are documented here.

## [Unreleased]

### Added
- Phase 0: package scaffold, pydantic configuration validation, deterministic hash-based splits, single-use certification ledger, typer CLI skeleton, and CI workflow.
- Phase 1: teacher clients for Groq and Gemini Decisions with proactive rate limits, backoff, resumable SQLite label cache, versioned prompt and strict parser, dry-run estimator, candidate selection/sanity gate, `.env` key loading, and `label` / `select-teacher` CLI commands.
- Phase 2: CPU-only `static_embed` and `minilm_onnx_int8`/`fp32` students, gold-label training, confidence calibration with a configured threshold grid, cascade evaluation against gold labels with cost/latency measurement, and `train-students` / `build-cascade` CLI commands.
- Phase 3: Learn-then-Test certification with Hoeffding-Bentkus p-values, single-start fixed-sequence testing, calibration-only candidate ordering, multiple-looks ledger correction, `certify` CLI, tests, and methodology docs.

### Results
- Gemini Decisions `gemini-decision-1` measured 0.765 accuracy against 200 gold calibration labels, with 0 invalid outputs, 158,963 input tokens, 0 output tokens, and 851.2 ms mean latency.
- On 300 certification examples, `static_embed` measured 0.837 accuracy with 11 teacher calls and 1.5 ms p95; `minilm_onnx_int8` measured 0.920 accuracy with 1 teacher call and 7.3 ms p95.
- Certification against the configured 95% target returned `INSUFFICIENT_EVIDENCE`: both cascades produced p-value 1.000 at alpha = delta = 0.05, and no certificate was claimed.
- Full details are in [docs/RESULTS.md](docs/RESULTS.md).
