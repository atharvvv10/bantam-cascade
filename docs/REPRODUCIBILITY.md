# Reproducibility

## One command, no keys

From the repository root after `pip install -e ".[dev]"`, with the git-ignored
frozen run artifacts present locally:

```bash
bantam-cascade reproduce --trials 200 --examples 600
```

This writes `runs/reproduction/`:

- `certificate_verification.json` — frozen SMS artifact rechecked without a new
  test look (config hash, split fingerprint/hash, p-value, ledger count);
- `plan.json` — certification sizes at the unchanged 95% target;
- `validity_boundary_below_target.json` and `validity_above_target.json` —
  200-trial resampling experiment with the naive control;
- `status_demo.json` — the three certification statuses (explicitly synthetic);
- `reproduction_manifest.json` — a summary and the commands for live runs.

`runs/` is git-ignored and is not shipped in the wheel or sdist, so a clean
checkout can regenerate the plan, validity, and status artifacts but cannot
verify the frozen certificate until the local `runs/sms-groq` artifacts (or
equivalent paths passed via `--certificate-config` and `--certificate-run-dir`)
are available. `reproduce` does **not** call a teacher API, re-label the
certification split, or re-run `certify`. It exits non-zero if the frozen
certificate fails any consistency check.

## Code gates

```bash
ruff check .
ruff format --check .
mypy
pytest -q
python scripts/scan_secrets.py --history
```

The release GIF is regenerated with `python scripts/make_demo_gif.py`; its text
comes only from the frozen result numbers and offline reports.

`scan_secrets.py` prints only pattern names, commits, and paths — never matched
values. On 2026-11-27 it reported zero key-shaped strings in tracked files, all
commits, and reflogs. Keys remain only in the git-ignored local `.env`.

## Live teacher-label runs

Live labeling needs your own provider key in `.env` and a config that pins the
provider, model id, prompt version, prices, and rate limits. Use a **fresh**
`--run-dir` and pre-register before certification. Commands are listed in
`reproduction_manifest.json`; the seed-43 recall run is the current example:

```bash
bantam-cascade label --config configs/sms-spam-recall-groq.yaml --run-dir runs/sms-recall-groq
bantam-cascade train-students --config configs/sms-spam-recall-groq.yaml --run-dir runs/sms-recall-groq
bantam-cascade build-cascade --config configs/sms-spam-recall-groq.yaml --run-dir runs/sms-recall-groq
bantam-cascade certify-recall --config configs/sms-spam-recall-groq.yaml --run-dir runs/sms-recall-groq
```

Never run `certify` on `runs/sms-groq`; its single look is recorded in the
ledger. A second look requires `--allow-reuse`, which divides delta by the
number of looks and is not a fresh certificate.
