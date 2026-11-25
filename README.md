# Bantam Cascade

*Bantam Cascade finds the cheapest small-CPU-model + big-LLM fallback cascade that provably meets a user's accuracy requirement, or says honestly that it cannot certify one.*

## Status: Phase 1 — teacher labeling

This repository is being built phase by phase. Phase 0 (scaffold, config, splits, ledger, CLI) and Phase 1 (Groq teacher client, cache, prompt/parser, dry-run estimator, selection) are complete. There are **no benchmark results yet** — every number that will eventually appear here must come from a results file produced by a real run. Until then, results are marked **TBD**.

## What the finished tool does

1. Takes a labeled text-classification dataset plus a target accuracy, confidence level, and p95 latency budget.
2. Trains small CPU-only students on teacher-LLM labels, calibrates them, and builds a cascade (student answers when confident, teacher otherwise).
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

Copy `.env.example` to `.env` in the project root and put your Groq key in it. The file is git-ignored; never commit or print the key.

```text
bantam-cascade label --config configs/banking77.yaml --dry-run
bantam-cascade label --config configs/banking77.yaml --limit 200
bantam-cascade select-teacher --config configs/banking77.yaml --limit 200
```

The shipped configs pin `openai/gpt-oss-20b` with the user's free-tier limits and prices of `0`; all prices are read from config only. Dry run makes no API calls and needs no key.

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
