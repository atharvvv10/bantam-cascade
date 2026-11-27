# Limitations

> Status: v0.1.1 revision. Measured values are in [RESULTS.md](RESULTS.md) and in
> the git-ignored run artifacts referenced there.

- The Learn-then-Test guarantee is finite-sample and conditional on i.i.d.
  certification examples from the same distribution as deployment. It says
  nothing about distribution shift.
- Accuracy is measured only against gold human labels. Teacher agreement is
  never used as accuracy; teacher mistakes and invalid teacher outputs count as
  losses when the fallback is taken.
- Teacher provider, model id, and prompt version are pinned by config and
  cached. If the provider changes behavior, the cached labels define the
  recorded result; API determinism is not guaranteed.
- Cost figures are estimates computed only from token counts and prices supplied
  by the user. The shipped configs use `0` prices, so reported costs are
  `$0.000000`; that is not a claim that real inference is free.
- Every cost estimate depends on pinned user-supplied prices; the tool does not
  fetch live prices.
- Students are CPU-only, single-label text classifiers. Multi-label
  classification, generation, and GPU pipelines are out of scope.
- The certification split is single-use. Repeated looks are blocked unless
  `--allow-reuse` is passed, in which case the ledger divides delta by the total
  number of looks.
- The SMS Spam certificate is for `ucirvine/sms_spam` with the pinned Groq
  teacher and the frozen `word_tfidf` student. It does not transfer to Banking77,
  other datasets, other teachers, or changed preprocessing.
- On Banking77 the measured cascades do not certify the configured 95% target;
  that result is `INSUFFICIENT_EVIDENCE`, not a certificate.
- Gemini Decisions blocked certification SMS messages under its
  prohibited-content policy. That was a provider/data mismatch; no certification
  outcome was derived from the blocked attempt.
- The exact binomial p-value is reported as a cross-check, but the pre-registered
  decision uses Hoeffding-Bentkus. Changing the test after seeing the result
  would be a new look and is not done here.
- The `status-demo` command uses synthetic evidence to show all three statuses.
  It is not a certificate or a real dataset result.
- The seed-43 SMS recall test has only 74 spam positives. At the pre-registered
  Bonferroni level, the conservative HB test requires zero missed spam messages;
  two misses produced `INSUFFICIENT_EVIDENCE` despite 0.993 accuracy. No recall
  certificate or relaxed target is claimed.
- In ablation runs that take fallbacks, `cascade_results.jsonl` adds cached live
  teacher latency to p95. Those p95 numbers are live-equivalent estimates, not
  cache-replay latency. The original certified SMS cascade took zero fallbacks.
- `bounded-distill`'s gold-label stress test is diagnostic only: it supplies
  gold labels where a deployed teacher would be, so it is not a deployable
  accuracy or cost claim.
- `tfidf_baseline` is the current name of the student formerly called
  `static_embed`. Old artifacts remain loadable through a legacy alias; historical
  results keep the old name in parentheses.
- Teacher-output licensing and data-use terms are summarized in
  [COMPLIANCE.md](COMPLIANCE.md). The repository and published package do not
  ship teacher labels, caches, or derived models; those stay in git-ignored
  `runs/`.
- Additional lightweight model probes were evaluated only on calibration data
  and did not clear the accuracy bar needed to justify another certification
  look. No certification data was used for that tuning.
