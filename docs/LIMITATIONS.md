# Limitations

> Status: Phase 3 complete. Measured values are in [RESULTS.md](RESULTS.md) and in the git-ignored run artifacts referenced there.

- The Learn-then-Test guarantee is finite-sample and conditional on i.i.d. certification examples from the same distribution as deployment. It says nothing about distribution shift.
- Accuracy is measured only against gold human labels. Teacher agreement is never used as accuracy; teacher mistakes and invalid teacher outputs count as losses when the fallback is taken.
- The teacher provider, model id, and prompt version are pinned by config and cached. If the provider changes behavior, the cached labels define the recorded result; API determinism is not guaranteed.
- Rate limits, prices, and model availability change. Every cost figure is an estimate computed only from token counts and prices supplied by the user.
- Students are CPU-only, single-label text classifiers. Multi-label classification, generation, and GPU pipelines are out of scope.
- The certification split is single-use. Repeated looks are blocked unless `--allow-reuse` is passed, in which case the ledger divides delta by the total number of looks.
- The current 300-example certification split is too small to certify the configured 95% target with the measured lightweight cascades. The result is `INSUFFICIENT_EVIDENCE`, not a certificate.
- Additional lightweight model probes were evaluated only on calibration data and did not clear the accuracy bar needed to justify another certification look. No certification data was used for that tuning.
