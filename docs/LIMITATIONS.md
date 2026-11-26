# Limitations

> Status: Phase 7 recorded. Measured values are in [RESULTS.md](RESULTS.md) and in the git-ignored run artifacts referenced there.

- The Learn-then-Test guarantee is finite-sample and conditional on i.i.d. certification examples from the same distribution as deployment. It says nothing about distribution shift.
- Accuracy is measured only against gold human labels. Teacher agreement is never used as accuracy; teacher mistakes and invalid teacher outputs count as losses when the fallback is taken.
- Teacher provider, model id, and prompt version are pinned by config and cached. If the provider changes behavior, the cached labels define the recorded result; API determinism is not guaranteed.
- Rate limits, prices, and model availability change. Every cost figure is an estimate computed only from token counts and prices supplied by the user.
- Students are CPU-only, single-label text classifiers. Multi-label classification, generation, and GPU pipelines are out of scope.
- The certification split is single-use. Repeated looks are blocked unless `--allow-reuse` is passed, in which case the ledger divides delta by the total number of looks.
- The SMS Spam certificate is for `ucirvine/sms_spam` with the pinned Groq teacher and the frozen `word_tfidf` student. It does not transfer to Banking77, other datasets, other teachers, or changed preprocessing.
- On Banking77 the measured lightweight cascades do not certify the configured 95% target; that result is `INSUFFICIENT_EVIDENCE`, not a certificate.
- Gemini Decisions blocked certification SMS messages under its prohibited-content policy. That was a provider/data mismatch; no certification outcome was derived from the blocked attempt.
- Additional lightweight model probes were evaluated only on calibration data and did not clear the accuracy bar needed to justify another certification look. No certification data was used for that tuning.
