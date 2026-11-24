# Limitations (draft)

> Status: Phase 0 scaffold. All measured values are **TBD** until produced by a real run.

- The guarantee is finite-sample and conditional on i.i.d. data and no distribution shift. It says nothing about deployment traffic that differs from the certified data.
- Accuracy is measured only against gold human labels. Teacher agreement is never used as accuracy.
- The teacher is pinned and cached; if the provider changes behavior, cached labels define the result.
- API rate limits, prices, and model availability change. Every cost figure is an ESTIMATE derived from user-supplied prices.
- Students are CPU-only and single-label text classification. Multi-label, generation, and GPU pipelines are out of scope.
- Repetitions in the validity experiment reuse a shared gold pool and are therefore not independent.
- Free-tier limits and model availability change; the exact teacher id and date used for any run are recorded in its results file.
