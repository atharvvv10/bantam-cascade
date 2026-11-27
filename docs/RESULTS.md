# Results

> Revision note (2026-11-27): this is the honest core-evidence record behind the
> v0.1.1 revision. Every number comes from a run artifact under `runs/` (git-ignored
> locally) or from a CLI command that can be re-run offline. The `0.95` target has
> not been changed. The original SMS certificate is untouched; no certification
> command was re-run against its split.

## Headline summary

| Question | Evidence | Result |
| --- | --- | --- |
| Does the cascade actually help? | `runs/core-results/cascade-benefit.json` | With 100 gold labels on SMS, student alone 0.947, teacher alone 0.890, cascade **0.970**. On Banking77 with 2,000 gold labels, `minilm_onnx_int8` alone and cascade both 0.897 with 0 fallbacks, while the small baseline cascade only reaches 0.823. |
| Is the statistical claim valid? | `runs/validity/*.json`, 200 resamples | Boundary-below-target false-certificate rate **0.000** for the valid test vs **0.040** for naive threshold picking (allowed 0.050). Above-target certification rate 0.810 (valid) vs 0.990 (naive). |
| How do prior tools compare? | `runs/baselines/*.json` | `tinydistill` MiniLM: 0.887 on the Banking77 certification split; its static model 0.850. `bounded-distill` on real Groq teacher labels found no useful SMS router at 95% agreement; a gold-label upper-bound stress test reached up to 0.983 with real Groq fallback. |
| Are all three statuses reachable? | `runs/status-demo/status_demo.json` | Synthetic logic demo shows `CERTIFIED`, `INSUFFICIENT_EVIDENCE`, and `NO_FEASIBLE_CONFIG` explicitly. Real Banking77 outcome is `INSUFFICIENT_EVIDENCE`. |
| Was the test set sized properly? | `bantam-cascade plan` | To certify 0.95 at 0.95 confidence assuming 0.98 true accuracy, the minimum is **240** certification examples; the shipped original run used 300. At assumed 0.97 accuracy it would need 529. |
| Is exact binomial available? | `runs/sms-groq/exact_binomial_analysis.json` | On the frozen SMS result (4 errors / 300), Hoeffding-Bentkus p = 0.001878 and exact binomial p = 0.000691; both reject at `delta = 0.05`. |
| Does recall also certify? | `runs/sms-recall-groq/recall_certification_result.json` | Fresh seed-43 split: accuracy 596/600 rejects, but spam recall 72/74 does not at the pre-registered `delta/2 = 0.025`. Status is **`INSUFFICIENT_EVIDENCE`**; no recall certificate is claimed. |

## 1. Cascade benefit: small alone, big alone, cascade

All rows use the same untouched 300-example certification split for the given
dataset. "Student" is trained on that many **gold human labels**. "Teacher" is a
cached, pinned API model. Accuracy is always against gold human labels, never
teacher agreement. The artifact is `runs/core-results/cascade-benefit.json`.

### SMS Spam, pinned Groq `openai/gpt-oss-20b`

| Gold labels for student | Student alone | Teacher alone | Cascade | Teacher calls | Fallback rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.947 | 0.890 | **0.970** | 35 | 0.117 |
| 200 | 0.937 | 0.890 | 0.967 | 28 | 0.093 |
| 300 | 0.947 | 0.890 | 0.967 | 58 | 0.193 |

This is the clearest cascade-help result: the big model is worse than the
small model on this task, but its errors are complementary. At 100 labels the
cascade gains 2.3 points over the student alone and 8.0 points over the teacher
alone while calling the teacher on only 11.7% of examples.

### Banking77, pinned Gemini Decisions `gemini-decision-1`

| Gold labels for student | Student alone | Teacher alone | Cascade | Teacher calls | Fallback rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.217 | 0.803 | 0.797 | 297 | 0.990 |
| 200 | 0.293 | 0.803 | 0.793 | 284 | 0.947 |
| 300 | 0.377 | 0.803 | 0.800 | 291 | 0.970 |
| 500 | 0.543 | 0.803 | 0.770 | 207 | 0.690 |
| 1,000 | 0.643 | 0.803 | 0.773 | 123 | 0.410 |
| 2,000 | 0.780 | 0.803 | 0.823 | 73 | 0.243 |
| 2,000 (`minilm_onnx_int8`) | **0.897** | 0.803 | **0.897** | 0 | 0.000 |

Here the small `tfidf_baseline` (formerly `static_embed`) is far weaker than the
teacher, and the cascade mostly tracks the teacher. At 2,000 labels the stronger
`minilm_onnx_int8` student beats the teacher by 9.4 points and needs no fallback
calls. We report this instead of hiding a case where the cascade does not help.

### Latency caveat on the 100-label SMS row

`runs/core-results/latency-note.json` documents why that row shows a ~2,580 ms
p95: Bantam's `cascade_results.jsonl` adds the **cached live teacher latency**
to each fallback example, so the number is a live-equivalent latency estimate,
not replay-from-cache latency. The cached teacher latencies had median 2,562 ms.
A standalone measurement of the saved student on the same machine gave p95
0.646 ms. The original certified SMS cascade took zero fallbacks and its 0.7 ms
p95 is unaffected.

## 2. Statistical validity: 200 resampled certification repetitions

Command: `bantam-cascade validity --trials 200 --examples 600`. Reports:
`runs/validity/boundary_below_target.json` and `runs/validity/above_target.json`.

| Scenario | Valid false certs | Naive false certs | Valid cert rate | Naive cert rate | Allowed false-cert rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| True accuracy just below 0.95 | **0.000** | **0.040** | 0.010 | 0.105 | 0.050 |
| True accuracy above 0.95 | 0.000 | 0.000 | 0.810 | 0.990 | 0.050 |

Interpretation: at the boundary the valid Learn-then-Test procedure did not
false-certify in 200 repetitions, while picking a threshold from the same test
data falsely certified 8 times out of 200. Above the target, the valid procedure
is more conservative (0.810 vs 0.990 certification rate) because it controls
family-wise error rather than rewarding repeated looks. This is a simulation of
the statistical procedure, not a new dataset certificate.

## 3. Baseline comparisons on the same data

Artifacts: `runs/baselines/tinydistill-banking77.json`,
`runs/baselines/bounded-distill-sms.json`, and
`runs/baselines/bounded-distill-sms-gold-upper.json`.

### Tinydistill

Published `tinydistill` Banking77 models were evaluated with
`Predictor("hf:<repo>")` on Bantam's 300-example Banking77 certification split.
No Bantam split, teacher label, or threshold was changed.

| Model | Correct / 300 | Accuracy |
| --- | ---: | ---: |
| `Shroomy/tinydistill-banking77-minilm` | 266 | 0.887 |
| `Shroomy/tinydistill-banking77-static` | 255 | 0.850 |

Where they win: the tinydistill MiniLM model beats Bantam's 2,000-label small
baseline (0.823) by 6.3 points. Where they do not: Bantam's own 2,000-label
MiniLM cascade reaches 0.897, and neither tinydistill model comes with the
finite-sample certificate that the SMS cascade has. Tinydistill is a
distillation tool, not a certification procedure; this is an accuracy
comparison only.

### Bounded-distill

The real-teacher run trained routers on Groq teacher labels for the SMS student
training texts and evaluated on the same 300 certification texts against gold.

| Setting | Coverage | Cascade accuracy vs gold | Note |
| --- | ---: | ---: | --- |
| `sgr` or `wilson`, target agreement 0.95/0.97/0.99 | 0.000 | 0.890 | Router always falls back to Groq; the teacher's gold accuracy is 0.890. |

Bounded-distill's own agreement bound is not reachable on these teacher labels,
so it refuses to cover any example. That is an honest conservative outcome, not
a bug in the comparison. A second **upper-bound stress test** supplied gold
human labels to bounded-distill as if they were teacher labels and fallback
labels. This is not deployable, but it shows what the router can do when the
labels are clean:

| Train labels | Method / target | Coverage | Cascade accuracy with gold fallback | Cascade accuracy with real Groq fallback |
| ---: | --- | ---: | ---: | ---: |
| 1,000 | `wilson` / 0.99 | 0.853 | 0.997 | 0.980 |
| 4,000 | `sgr` / 0.99 | 0.930 | 0.993 | 0.983 |
| 4,000 | `wilson` / 0.99 | 0.957 | 0.987 | 0.980 |

This is where bounded-distill wins: with clean labels it produces a high-coverage
router that beats the teacher-alone 0.890. With the real teacher labels it does
not, which is why Bantam's headline result still comes from its own
calibration-selected threshold and single-use test.

## 4. Who does the small model learn from?

Bantam's small students **train only on gold human labels**. The teacher is used
only as a cascade fallback and is separately scored against gold. The ablation
above holds the teacher fixed and varies the number of gold labels.

The baselines make the contrast explicit:

| Tool | Student training labels | Fallback labels | Deployable? |
| --- | --- | --- | --- |
| Bantam (`word_tfidf`, `tfidf_baseline`, `minilm_onnx_int8`) | Gold human labels only | Cached pinned teacher | Yes |
| `bounded-distill` real-teacher run | Teacher predictions | Teacher predictions | Yes, but coverage 0 here |
| `bounded-distill` gold upper-bound stress test | Gold labels supplied as teacher labels | Gold (stress test) or real Groq (fallback measurement) | No — diagnostic only |

## 5. All certification statuses

`bantam-cascade status-demo --output-dir runs/status-demo` writes
`runs/status-demo/status_demo.json`. It is explicitly marked
`"synthetic": true` and `"not_a_certificate": true`.

| Expected status | Actual status | Errors in synthetic case |
| --- | --- | ---: |
| `CERTIFIED` | `CERTIFIED` | 4 / 300 |
| `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | 24 / 300 |
| `NO_FEASIBLE_CONFIG` | `NO_FEASIBLE_CONFIG` | 4 / 300 (p95 500 ms vs 200 ms budget) |

Real-world status summary:

| Dataset | Status | Source |
| --- | --- | --- |
| SMS Spam, pinned Groq, original frozen split | `CERTIFIED` | `runs/sms-groq/certification_result.json` |
| Banking77, pinned Gemini | `INSUFFICIENT_EVIDENCE` | `runs/phase2-gemini/certification_result.json` |
| `NO_FEASIBLE_CONFIG` | Not claimed on real data; demonstrated synthetically | `runs/status-demo/status_demo.json` |

## 6. Certification split sizing

Command: `bantam-cascade plan --target-accuracy 0.95 --confidence-level 0.95 --assumed-accuracy <a>`.

| Assumed true accuracy | Minimum certification examples |
| ---: | ---: |
| 0.970 | 529 |
| 0.975 | 349 |
| 0.980 | 240 |
| 0.985 | 181 |
| 0.990 | 150 |

The original SMS certificate used 300 examples, which is between the 0.98 and
0.985 planning rows. The plan command is a pre-test sizing tool; it is not a
certificate.

## 7. Co-primary accuracy and spam recall (fresh split)

Config: `configs/sms-spam-recall-groq.yaml`. This is a fresh seed-43 split, not
a second look at the frozen seed-42 certificate. The pre-registration in
`runs/sms-recall-groq/preregistration.json` was written before certification
labels were requested. It fixed the 600-example split, positive label `spam`,
Bonferroni `delta/2 = 0.025` per endpoint, 200 ms p95 budget, and candidate
order. The original frozen certificate is untouched.

| Item | Value |
| --- | --- |
| Split sizes (student / calibration / certification) | 4,000 / 200 / 600 |
| Spam positives in certification | 74 |
| Calibration accuracy vs gold | 0.990 student, 0.900 teacher |
| Frozen threshold from calibration | 0.612 |
| Teacher fallback calls | 3 / 600 (0.005) |
| Accuracy vs gold | 0.993 (4 errors / 600) |
| Accuracy HB / exact p | 5.476e-09 / 2.015e-09 |
| Spam recall vs gold | 0.973 (2 errors / 74) |
| Recall HB / exact p | 0.613 / 0.278 |
| p95 latency | 1.7 ms |
| Prior looks | 0 |
| Status | **`INSUFFICIENT_EVIDENCE`** |

The accuracy endpoint clearly rejects the 5% risk null, but spam recall does
not. With only 74 positive examples and a Bonferroni-split level of 0.025, the
conservative Hoeffding-Bentkus test needs zero missed spam messages; this run
missed two. We do **not** lower the recall target, swap tests after seeing the
result, or claim a recall certificate. Source:
`runs/sms-recall-groq/recall_certification_result.json`.

## 8. Original certified benchmark (unchanged)

Config: `configs/sms-spam-groq.yaml`. Student: `word_tfidf`. Teacher: pinned
Groq `openai/gpt-oss-20b`, prompt `v1`.

| Item | Value |
| --- | --- |
| Student train / calibration / certification | 4,000 / 200 / 300 |
| Student calibration accuracy | 0.995 |
| Teacher calibration accuracy vs gold | 0.945 |
| Selected threshold | 0.000 (student-only; 0 fallback calls) |
| Measured certification accuracy | 0.987 (4 errors / 300) |
| Hoeffding-Bentkus p-value | 0.001878 |
| Exact binomial p-value (derived, not a new test) | 0.000691 |
| Alpha / delta | 0.050 / 0.050 |
| Prior looks | 0 |
| p95 latency | 0.7 ms |
| Status | **`CERTIFIED`** |

Source: `runs/sms-groq/certification_result.json`. Derived comparison:
`runs/sms-groq/exact_binomial_analysis.json`. The certificate is conditional on
the assumptions in [METHODOLOGY.md](METHODOLOGY.md).

Gemini Decisions was attempted first on SMS but blocked certification text under
`code: prohibited_content`; no certification labels or outcome came from that
attempt. The Groq run is the only SMS certificate.

## 9. Uncertified benchmark: Banking77

Config: `configs/banking77-gemini.yaml`. Teacher: pinned Gemini Decisions
`gemini-decision-1`, prompt `v1`.

| Student (current name) | Errors / 300 | Accuracy | HB p-value | Status |
| --- | ---: | ---: | ---: | --- |
| `minilm_onnx_int8` | 24 | 0.920 | 1.000 | not rejected |
| `tfidf_baseline` (formerly `static_embed`) | 49 | 0.837 | 1.000 | not rejected |

Outcome: `INSUFFICIENT_EVIDENCE`. No certificate is claimed. Source:
`runs/phase2-gemini/certification_result.json`.

## Not run

`clinc150` was not run. No accuracy, cost, latency, or certificate is claimed for it.
