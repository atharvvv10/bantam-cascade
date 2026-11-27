# Changelog

All notable changes to Bantam Cascade are documented here.

## [Unreleased]

### Changed
- Pinned `space/requirements.txt` to tag `v0.1.1` and refreshed the README
  status line after publication.
- Made the Space scaffold self-contained with the frozen `word_tfidf` threshold
  row and its gold-label-trained student artifact; no teacher labels, caches,
  keys, or certificates are included.
- Kept the Space on the Gradio SDK. `app.py` imports the pinned `v0.1.1` source
  tag from `src/` when the package is not installed, and requirements list only
  the student inference packages. This avoids the released package's pinned
  `pydantic==2.14.4` conflicting with the HF Gradio SDK's `pydantic<=2.12.5`.

## [0.1.1] - 2026-11-27

The published `v0.1.0` artifacts are unchanged. This revision adds evidence and
tooling; no target, threshold, or goalpost is moved.

### Added
- Cascade-benefit artifacts comparing student alone, teacher alone, and cascade
  across gold-label budgets on SMS Spam and Banking77.
- 200-trial Learn-then-Test validity experiment with a naive-threshold control,
  plus exact-binomial cross-checks, split-sizing via `plan`, a synthetic demo of
  all three certification statuses, and honest `tinydistill` / `bounded-distill`
  baselines on the same data.
- `certify-recall` for pre-registered co-primary accuracy and positive-class
  recall with a Bonferroni-split family-wise error budget; the fresh seed-43
  SMS run passes accuracy but returns `INSUFFICIENT_EVIDENCE` for recall.
- `verify-certificate` and one-command `reproduce` for offline verification of
  frozen artifacts without a new certification look.
- `docs/COMPLIANCE.md` with provider provenance and a conservative licensing
  position; `scripts/scan_secrets.py` for key-shaped strings in tracked files
  and git history.
- CI and PyPI badges, a CLI-evidence GIF, an offline Gradio Space scaffold,
  `docs/BLOG_POST.md`, repository description/topics, and a manual TestPyPI
  publishing workflow.

### Changed
- Renamed `static_embed` to `tfidf_baseline`; old configs and artifacts remain
  loadable through a documented legacy alias.
- Softened the README tagline from "provably meets" to describe the actual
  finite-sample statistical test and its assumptions.
- Results, methodology, and limitations docs now report where the cascade does
  and does not help, including cached-latency and unpaid-quota caveats.

### Published
- TestPyPI run `45950924942` and PyPI run `45951719762` published `0.1.1` with
  Trusted Publishing; no long-lived token was created or stored.
- The `v0.1.1` GitHub Release contains the exact PyPI artifacts, byte for byte.
- Clean-venv installs from both indexes passed `--version`, `plan`, `validity`,
  and `status-demo`.

## [0.1.0] - 2026-11-26

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

### Phases 4–7: real 95% certificate
- Added a calibration-only benchmark probe across five free datasets; selected `ucirvine/sms_spam` after word TF-IDF reached 98.5% calibration accuracy without touching certification data.
- Added the CPU-only `word_tfidf` student and namespaced SMS Spam dataset support, with offline tests.
- Added pre-registration before the certification look: config hash, split fingerprint, frozen threshold, and decision rule.
- Gemini Decisions blocked certification SMS messages under its prohibited-content policy; that attempt produced no certification labels and was discarded.
- Pivoted to the pinned Groq `openai/gpt-oss-20b` teacher with a fresh run and fresh pre-registration. The calibration-only threshold search selected student-only prediction (threshold 0.000).
- Certification on 300 held-out SMS Spam examples returned `CERTIFIED`: 4 errors / 300 (0.987 measured accuracy), p-value 0.001878 at alpha = delta = 0.05, 0 fallback calls, $0.000000 estimated cost, 0.7 ms p95.
- Fixed a reporting inconsistency where rejected candidates were serialized with `certified: false` even though they appeared in `certified` and `chosen`.
- Gemini client errors now include a truncated response body for diagnosis; the API key is still never logged.

### Phases 8–11: verification, release, demo, and hardening
- Verified the frozen config hash and split fingerprint against pre-registration, confirmed zero prior looks, recomputed the Hoeffding-Bentkus p-value, and checked that the certificate would survive one extra look.
- Added an offline `demo` command that classifies one text from saved student artifacts without API calls and without re-running certification.
- Added a regression test proving Gemini HTTP errors include a truncated response body but never the key.
- Bumped the package, citation, and runtime version to 0.1.0; included the shipped configs and docs in the sdist; built the sdist and wheel; passed `twine check`; and verified no `.env` or run artifacts are packaged.
- Final gates: `ruff check .`, `ruff format --check .`, `mypy`, and 96 passing `pytest` tests.

### Release automation
- Added a pinned, OIDC-based PyPI publishing workflow that builds, checks, and uploads distributions without a long-lived API token.
- Added `docs/RELEASING.md` with the one-time PyPI Trusted Publisher setup and publish instructions.
