# Results

> Status: Phase 7 recorded. Every number below comes from a real run artifact under `runs/` (git-ignored locally). The 95% target is **certified on SMS Spam** and **not certified on Banking77**. Both outcomes are reported honestly.

## Certified benchmark: SMS Spam

Config: `configs/sms-spam-groq.yaml`. Student: `word_tfidf` (word n-gram TF-IDF + logistic regression, CPU-only, no model downloads). Teacher: pinned Groq `openai/gpt-oss-20b`, prompt `v1`.

Pre-registration: `runs/sms-groq/preregistration.json` was written before any certification text was read. It freezes the config hash, split fingerprint, threshold grid, selected threshold, and decision rule.

| Item | Value |
| --- | --- |
| Dataset | `ucirvine/sms_spam` (`plain_text`) |
| Student train / calibration / certification | 4,000 / 200 / 300 |
| Student calibration accuracy | 0.995 |
| Teacher calibration accuracy vs gold | 0.945 (0 invalid, 35,806 in / 7,673 out tokens, 1,985.3 ms mean) |
| Selected threshold | 0.000 (student-only; teacher is weaker than student) |
| Calibration cascade accuracy | 0.995 (0 fallback calls) |
| Measured certification accuracy | 0.987 (4 errors / 300, 0 fallback calls) |
| Estimated cost | $0.000000 from configured prices |
| p95 certification latency | 0.7 ms |
| Alpha / delta | 0.050 / 0.050 |
| Hoeffding-Bentkus p-value | 0.001878 |
| Prior looks | 0 |
| Status | **`CERTIFIED`** |

Source: `runs/sms-groq/certification_result.json`. The reported certificate is conditional on the assumptions in [METHODOLOGY.md](METHODOLOGY.md).

Why the fallback was not used: the pinned Groq teacher is weaker than the student on calibration (0.945 vs 0.995), so the calibration-only threshold search selected 0.000, meaning the student answers every example. The teacher was still run on the certification split for complete measurement, but the cascade took zero fallback calls.

Gemini Decisions was attempted first but blocked one or more certification SMS messages under its prohibited-content policy (`code: prohibited_content`). No certification labels or outcomes were produced from that attempt, and it was discarded before the Groq run.

## Uncertified benchmark: Banking77

Config: `configs/banking77-gemini.yaml`. Teacher: pinned Gemini Decisions `gemini-decision-1`, prompt `v1`.

| Dataset | Teacher | Examples | Accuracy vs gold | Invalid | Input tokens | Output tokens | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | Gemini Decisions `gemini-decision-1` | 200 | 0.765 | 0 | 158,963 | 0 | 851.2 ms |

Source: `runs/phase1-gemini/teacher_labels.jsonl`. A Groq `openai/gpt-oss-20b` sanity check on the same 200 gold labels measured 0.745 accuracy with 0 invalid outputs.

Measured cascades on 300 certification examples with the Gemini fallback (`runs/phase2-gemini/cascade_results.jsonl`):

| Student | Accuracy | Fallback rate | Teacher calls | Est. cost | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `static_embed` | 0.837 | 0.037 | 11 | $0.000000 | 1.5 ms |
| `minilm_onnx_int8` | 0.920 | 0.003 | 1 | $0.000000 | 7.3 ms |

Certification against the configured 95% target (`runs/phase2-gemini/certification_result.json`):

| Item | Value |
| --- | --- |
| `minilm_onnx_int8` | 24 errors / 300, p-value 1.000, not rejected |
| `static_embed` | 49 errors / 300, p-value 1.000, not rejected |
| Chosen cascade | none |
| Status | `INSUFFICIENT_EVIDENCE` |

Neither Banking77 cascade rejects the null that its true error rate exceeds 5%. No certificate is claimed on Banking77.

## Not run

`clinc150` was not run. No accuracy, cost, latency, or certificate is claimed for it.
