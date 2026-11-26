# Results

> Status: Phase 3 complete. Every number below comes from a real run artifact under `runs/` (git-ignored locally). The 95% target is **not certified**; the honest status is `INSUFFICIENT_EVIDENCE`.

## Teacher sanity check

| Dataset | Teacher | Examples | Accuracy vs gold | Invalid | Input tokens | Output tokens | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | Gemini Decisions `gemini-decision-1`, prompt `v1` | 200 | 0.765 | 0 | 158,963 | 0 | 851.2 ms |

Source: `runs/phase1-gemini/teacher_labels.jsonl`. A Groq `openai/gpt-oss-20b` sanity check on the same 200 gold labels measured 0.745 accuracy with 0 invalid outputs. Both are measurements against gold labels, not certificates.

## Cascade measurements

Source: `runs/phase2-gemini/cascade_results.jsonl`, evaluated on 300 certification examples with the Gemini fallback.

| Student | Accuracy | Fallback rate | Teacher calls | Est. cost | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `static_embed` | 0.837 | 0.037 | 11 | $0.000000 | 1.5 ms |
| `minilm_onnx_int8` | 0.920 | 0.003 | 1 | $0.000000 | 7.3 ms |

These are measured cascades, not statistical certificates.

## Certification

Source: `runs/phase2-gemini/certification_result.json`.

| Item | Value |
| --- | --- |
| Target accuracy | 0.950 |
| Confidence level | 0.950 |
| Alpha / delta | 0.050 / 0.050 |
| Prior looks | 0 |
| `minilm_onnx_int8` | 24 errors / 300, p-value 1.000, not rejected |
| `static_embed` | 49 errors / 300, p-value 1.000, not rejected |
| Chosen cascade | none |
| Status | `INSUFFICIENT_EVIDENCE` |

Neither cascade rejects the null that its true error rate exceeds 5%. No certified accuracy lower bound is claimed.

## Not run

`clinc150` was not run. No accuracy, cost, latency, or certificate is claimed for it.
