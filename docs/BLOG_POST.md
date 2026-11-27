# What Bantam Cascade actually shows

*An honest one-page summary of the 0.1.1 revision. Not legal advice, not a
general claim about LLM cascades.*

## The question

Can a tiny CPU text classifier plus occasional big-model fallback meet a 95%
accuracy target with a statistical guarantee, without fine-tuning a large
model? And can we say so honestly?

## The honest answer: sometimes, under stated assumptions

On `ucirvine/sms_spam`, a `word_tfidf` student trained on 4,000 gold human
labels was tested once on 300 untouched examples. The pre-registered cascade
selected a student-only threshold (0 fallback calls) and scored 4 errors /
300 = 0.987 accuracy against gold. The Hoeffding-Bentkus p-value was 0.001878 at
alpha = delta = 0.05, with zero prior looks. That is a real finite-sample
result. It is conditional on i.i.d., deployment-matched data and a pinned,
cached teacher; it says nothing about drift.

On Banking77 the same machinery returns `INSUFFICIENT_EVIDENCE` at the same 95%
target. We do not dress that up as a success.

## Does the cascade help?

Yes on SMS, for a clear reason: the teacher is *worse* than the student overall
(0.890 vs 0.947 with 100 gold labels), but its errors are complementary. The
cascade reached 0.970 while calling the teacher on 11.7% of examples. With more
gold labels the gap shrinks; with 300 labels the cascade reached 0.967.

No on Banking77 with a stronger student: `minilm_onnx_int8` trained on 2,000
gold labels scored 0.897 alone, beat the 0.803 teacher, and needed 0 fallbacks.
The small TF-IDF baseline cascade reached only 0.823. Reporting only the win
would have been misleading.

We also tried a fresh, pre-registered test of accuracy **and** spam recall on a
600-message seed-43 split. Accuracy was 596/600, but recall was 72/74, and with
only 74 positive examples the Bonferroni-split bound requires zero misses. The
result is `INSUFFICIENT_EVIDENCE`. We did not lower the recall target.

## Is the statistics sound?

We ran 200 fresh resamples of a known population just below the 0.95 target.
The Learn-then-Test procedure false-certified 0/200; picking a threshold on the
test data itself false-certified 8/200 (allowed: 5%). Above the target the valid
procedure certified 81% of the time versus 99% for the naive control. The cost
of honesty is conservatism, not magic.

An exact binomial cross-check on the frozen SMS counts gives p = 0.000691;
Hoeffding-Bentkus gives 0.001878. Both reject at 0.05. The pre-registered test
remains Hoeffding-Bentkus; swapping tests after seeing the result would be a
new look.

## What other tools did

`tinydistill`'s published Banking77 MiniLM model scored 0.887 on the same 300
texts, beating Bantam's 2,000-label TF-IDF cascade (0.823) but below Bantam's
own MiniLM cascade (0.897). `bounded-distill` refused to cover *any* SMS example
when given the real Groq teacher labels because its agreement target was
unreachable; that is a conservative, correct outcome. In a diagnostic stress
test with gold labels supplied as teacher labels it reached up to 0.983 with
real Groq fallback — useful, but not deployable and not a Bantam result.

## What we changed after the first release

We did not lower the target, move thresholds, or re-test the original
certificate. We added:

- a cascade-benefit table with small-alone / big-alone / cascade rows;
- the 200-trial validity experiment;
- `plan` sizing (240 certification examples at assumed 0.98 accuracy; the
  original run used 300);
- a synthetic demo of `CERTIFIED`, `INSUFFICIENT_EVIDENCE`, and
  `NO_FEASIBLE_CONFIG`;
- co-primary accuracy-and-recall certification with a Bonferroni error budget;
- `verify-certificate` and one-command `reproduce`;
- licensing/provenance notes and a keys-in-history scan.

`static_embed` was renamed `tfidf_baseline`; old configs still load.

## Caveats we will not hide

Costs are estimates from user-supplied prices, which are zero in the shipped
configs: `$0.000000` is not a claim that inference is free. Some ablations show
~2.5 s p95 because cached live teacher latency is added to fallback examples;
a direct student measurement was ~0.646 ms. The original certified cascade took
zero fallbacks. Gemini-derived exploratory runs lack paid-quota provenance, so
we do not publish those labels or models. Teacher outputs remain local and
git-ignored.

## Bottom line

Bantam Cascade is a small, honest risk-control tool, not a magic accuracy
machine. It finds a real certificate on one dataset, refuses to claim one on
another, and now ships the evidence needed to check both claims.
