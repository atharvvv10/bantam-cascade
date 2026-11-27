# Methodology

Bantam Cascade uses a learn-then-test design. A calibration split chooses a
threshold; an untouched certification split tests it once. A result is called
**certified** only when the pre-registered test rejects the null at the
configured error level. Otherwise the CLI reports `INSUFFICIENT_EVIDENCE` or
`NO_FEASIBLE_CONFIG` and exits non-zero.

The target is always **0.95 accuracy** unless the user explicitly changes it in
a config. No threshold, dataset, or endpoint was moved to make a result pass.

## Data and loss

- `student_train` examples train each CPU student on **gold human labels**.
- `calibration` examples select one confidence threshold per student and define
  the pre-registered certification order.
- `certification` examples are used once for testing. The SQLite ledger records
  every look.
- A cascade returns the student label when student confidence is at least `t`,
  otherwise the pinned teacher label.
- The per-example loss is `1` when the **returned** label differs from the
  **gold human** label. Teacher mistakes and invalid teacher outputs count as
  losses when the fallback is taken.
- Cost is not a statistical quantity. Every cost is an estimate computed only
  from token counts and prices in the user's config.

## Learn step

`select_threshold` searches a pre-registered grid on the calibration split and
picks the threshold with the best measured cascade accuracy (ties broken toward
fewer teacher calls). The threshold is fixed before certification. The
certification order is also learned from calibration data only: calibration
accuracy descending, then estimated cost, fallback rate, and student name. No
certification measurement changes the order.

## Test step

Let `alpha = 1 - minimum_accuracy` and `delta = 1 - confidence_level`. For
candidate `j`, let `n` be the certification size, `k` the observed losses, and
`R_hat = k / n`. The null is `H_j: risk(j) > alpha`.

The Hoeffding-Bentkus p-value is

```text
h1(a, b) = a log(a / b) + (1 - a) log((1 - a) / (1 - b))

p_j = min(
    exp(-n * h1(min(R_hat, alpha), alpha)),
    min(1, e * P(Bin(n, alpha) <= ceil(n * R_hat)))
)
```

The exact one-sided binomial p-value `P(Bin(n, alpha) <= k)` is also computed and
reported in the `certify` table. It is not substituted for the pre-registered
Hoeffding-Bentkus decision; it is a transparent cross-check. For the frozen SMS
result (4 errors / 300), HB p = 0.001878 and exact binomial p = 0.000691.

A single-start fixed-sequence test walks the pre-registered order and stops at
the first p-value above `delta`. The rejected set is certified with family-wise
error rate at most `delta`. Learn-then-Test permits choosing any rejected
configuration using the same test data; we choose the lowest estimated cost,
then lowest fallback rate, among rejected candidates that also meet the measured
p95 latency budget.

## Single-use split and repeated looks

A SQLite ledger records split hash, candidate thresholds, and delta for every
certification attempt. A second attempt is refused unless `--allow-reuse` is
passed; with that flag, `delta` is divided by the total number of looks,
preserving the level under repeated testing.

## Sizing a certification split before testing

`bantam-cascade plan` searches for the smallest `n` such that a conservative
rounded-up error count at an assumed true accuracy rejects
`risk > 1 - target`. With target 0.95 and confidence 0.95, the required sizes
are 529 at assumed 0.97, 349 at 0.975, 240 at 0.98, 181 at 0.985, and 150 at
0.99. This is a planning number, not a certificate.

## Validity experiment

`bantam-cascade validity` simulates 200 fresh samples from a known population
and applies the same fixed-sequence logic. A naive control picks the best
threshold on the test split itself. Reports in `runs/validity/`:

- boundary-below-target (true accuracy just below 0.95): valid false-certificate
  rate 0.000, naive 0.040, allowed 0.050;
- above-target: valid certification rate 0.810, naive 0.990.

The naive control's higher false-certificate rate is the empirical cost of
ignoring the pre-registration/multiple-looks correction.

## Co-primary accuracy and recall

For imbalanced data such as SMS Spam, `certify-recall` tests overall accuracy
and positive-class recall as co-primary endpoints. Each is tested at
`delta / 2` using the same Hoeffding-Bentkus p-value, and the fixed sequence
stops at the first candidate that fails either endpoint. The pre-registration
freezes the positive label, split hash, config hash, and Bonferroni split before
any certification label is read. Recall is computed against gold human labels.
With few positive examples and a Bonferroni-split level, the HB bound can require
zero missed positives; that is a pre-registered power consequence, not a reason
to relax the target after the result.

## Assumptions printed with any certified claim

- certification examples are i.i.d. draws from the deployment distribution;
- deployment distribution matches the calibration and certification data used here;
- teacher model id, date, and prompt version are pinned and cached;
- API determinism is not guaranteed, so results are relative to cached teacher labels;
- accuracy (and recall, where tested) is measured only against gold human labels;
- latency is measured per example and p95 is compared to the configured budget.

A certificate says nothing about shifted deployment data, a changed teacher
model, or unverified teacher labels.

## Verification without another look

`bantam-cascade verify-certificate` rechecks config hash, split fingerprint and
hash, threshold, error count, recomputed p-values, delta, latency budget, and
ledger count from stored artifacts. It does not re-label the certification
split or call `certify`. `bantam-cascade reproduce` runs that verification plus
the pre-test size plan, validity simulation, and status demonstration in one
command; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Current measured outcomes

See [RESULTS.md](RESULTS.md) for the full tables.

- **SMS Spam, pinned Groq teacher, `word_tfidf`:** 4 errors / 300, HB p 0.001878
  at `alpha = delta = 0.05`, zero prior looks, `CERTIFIED`. The threshold was
  0.000 (student-only) and frozen in `runs/sms-groq/preregistration.json`.
- **Banking77, pinned Gemini teacher:** 24 errors / 300 for
  `minilm_onnx_int8` and 49 / 300 for `tfidf_baseline`, both p-values 1.000,
  `INSUFFICIENT_EVIDENCE`. No certificate is claimed.
- **SMS Spam, fresh seed-43 recall test:** accuracy 4 errors / 600 rejects at
  `delta/2`, but spam recall 2 errors / 74 does not (HB p 0.613), so the
  co-primary outcome is `INSUFFICIENT_EVIDENCE`. This is a separate
  pre-registered split, not a new look at the frozen certificate.
- **Gemini Decisions model id:** `gemini-decision-1` is the documented model code
  for the Decisions API (`POST https://generativelanguage.googleapis.com/v1beta/decisions`).
  The v1beta discovery document lagged the published docs when checked on
  2026-11-27; the config correctly uses the documented id and does not switch to
  a generative model id.

## Prior work

Bantam Cascade does not claim novelty for distillation, cascades, or threshold
selection with guarantees. It builds on tinydistill (MIT), bounded-distill
(Apache-2.0), FrugalGPT (arXiv:2305.05176), and Learn then Test (Angelopoulos,
Bates, Candès, Jordan, Lei; arXiv:2110.01052, DOI 10.1214/24-AOAS1998). See
`CITATION.cff`.
