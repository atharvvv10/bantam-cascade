# Methodology

Bantam Cascade uses a learn-then-test design. The learning split chooses a threshold; the single-use certification split tests it. A result is called **certified** only when the test rejects the null hypothesis at the configured error level. Otherwise the CLI reports `INSUFFICIENT_EVIDENCE` or `NO_FEASIBLE_CONFIG` and exits non-zero.

## Data and loss

- `student_train` examples train each CPU student on **gold human labels**.
- `calibration` examples (200 in the shipped configs) select one confidence threshold per student and define the certification order.
- `certification` examples (300 in the shipped configs) are used once for testing.
- A cascade returns the student label when student confidence is at least `t`, otherwise the pinned teacher label.
- The per-example loss is `1` when the **returned** label differs from the **gold human** label. Teacher mistakes and invalid teacher outputs count as losses.
- Cost is not a statistical quantity. Every cost is an estimate computed only from token counts and prices in the user's config.

## Learn step

For each student, `select_threshold` searches a pre-registered grid on the calibration split and picks the threshold with the best measured cascade accuracy (ties broken toward fewer teacher calls). The threshold is fixed before certification. The certification order is also learned from calibration data only: calibration accuracy descending, then estimated cost, fallback rate, and student name. No certification measurement changes the order.

## Test step

Let `alpha = 1 - minimum_accuracy` and `delta = 1 - confidence_level`. For candidate `j`, let `n` be the number of certification examples, `k` the number of observed losses, and `R_hat = k / n`. The null is `H_j: risk(j) > alpha`.

The Hoeffding-Bentkus p-value is

```text
h1(a, b) = a log(a / b) + (1 - a) log((1 - a) / (1 - b))

p_j = min(
    exp(-n * h1(min(R_hat, alpha), alpha)),
    min(1, e * P(Bin(n, alpha) <= ceil(n * R_hat)))
)
```

This is finite-sample valid for bounded `[0, 1]` losses. The binomial CDF is computed with a recurrence so no large binomial coefficients are materialized.

A single-start fixed-sequence test walks the pre-registered order and stops at the first p-value above `delta`. The rejected set is certified with family-wise error rate at most `delta`. Learn-then-Test permits choosing any rejected configuration using the same test data; we choose the lowest estimated cost, then lowest fallback rate, among rejected candidates that also meet the measured p95 latency budget.

## Single-use split and repeated looks

A SQLite ledger records `(split hash, candidate thresholds, delta)` for every certification attempt. The split hash is computed from the certification texts. A second attempt is refused unless `--allow-reuse` is passed; with that flag, `delta` is divided by the total number of looks (`delta / (prior_looks + 1)`), preserving the level under repeated testing.

## Assumptions printed with any certified claim

- certification examples are i.i.d. draws from the deployment distribution;
- deployment distribution matches the calibration and certification data used here;
- teacher model id, date, and prompt version are pinned and cached;
- API determinism is not guaranteed, so results are relative to cached teacher labels;
- accuracy is measured only against gold human labels.

A certificate says nothing about shifted deployment data, a changed teacher model, or unverified teacher labels.

## Current measured outcomes

Two real runs exist; see [RESULTS.md](RESULTS.md) for the full tables.

- **SMS Spam, pinned Groq teacher, `word_tfidf` student:** 4 errors / 300, p-value 0.001878 at `alpha = delta = 0.05`, zero prior looks, `CERTIFIED`. The threshold was 0.000 (student-only) and was frozen in `runs/sms-groq/preregistration.json` before certification data was read.
- **Banking77, pinned Gemini teacher:** 24 errors / 300 for `minilm_onnx_int8` and 49 / 300 for `static_embed`, both p-values 1.000 at `delta = 0.05`, `INSUFFICIENT_EVIDENCE`. No certificate is claimed there.

A Gemini Decisions attempt on SMS Spam was blocked by the provider's prohibited-content policy before any certification labels were produced, so it is not part of either outcome.

## Prior work

Bantam Cascade does not claim novelty for distillation, cascades, or threshold selection with guarantees. It builds on tinydistill (MIT), bounded-distill (Apache-2.0), FrugalGPT (arXiv:2305.05176), and Learn then Test (Angelopoulos, Bates, Candès, Jordan, Lei; arXiv:2110.01052, DOI 10.1214/24-AOAS1998). See `CITATION.cff`.
