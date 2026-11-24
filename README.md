# Bantam Cascade

*Bantam Cascade finds the cheapest small-CPU-model + big-LLM fallback cascade that provably meets a user's accuracy requirement, or says honestly that it cannot certify one.*

> **Naming note:** This project is unrelated to the existing [`bantam`](https://pypi.org/project/bantam/) microservices package on PyPI. Our import package is therefore always `bantam_cascade`, never `bantam`. The PyPI distribution and CLI are both named `bantam-cascade`; the GitHub repository is `bantam-cascade`.

## Status: early scaffolding

This repository is being built phase by phase. There are **no benchmark results yet** — every number that will eventually appear here must come from a results file produced by a real run. Until then, results are marked **TBD**.

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

## How certification works

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the plain-English explanation and the exact statistical assumptions.

## Limitations

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md). Nothing here is a guarantee about deployment data that differs from the certified data.

## Prior work and credits

Bantam Cascade does **not** claim novelty for distillation, cascades, or threshold selection with guarantees. It builds on TinyDistill, bounded-distill, FrugalGPT, and Learn-then-Test (Angelopoulos, Bates, Candès, Jordan, Lei). Full citations are in [CITATION.cff](CITATION.cff) and [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
