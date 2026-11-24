# Methodology (draft)

> Status: Phase 0 scaffold. Statistical implementation arrives in Phase 3. No results numbers are recorded here yet; all such values are **TBD** until produced by a real run.

## Plain-English summary (planned)

Bantam Cascade trains small CPU-only students on teacher-LLM labels, calibrates their confidence on gold-labeled calibration data, and considers a cascade: the student answers if its confidence is at least a threshold `t`, otherwise the teacher answers. A per-example loss is 1 when the **returned** label differs from the **gold human** label, including teacher mistakes and invalid teacher outputs.

A threshold grid is pre-registered using calibration data only. Certification then uses Learn-then-Test fixed-sequence testing on a single-use certification split. Every rejected hypothesis is certified with family-wise error rate at most `delta`; the guarantee covers the whole cascade, not just the student.

The threshold must never be chosen and certified on the same data: doing so would invalidate the guarantee. Multiple candidates share `delta` by Bonferroni.

## Assumptions that must be printed with any "certified" claim

- certification examples are i.i.d. draws from the deployment distribution;
- deployment distribution matches the data used for calibration and certification;
- teacher model id, date, and prompt version are pinned and cached;
- API determinism is not guaranteed, so results are relative to cached labels.

## Prior work

Bantam Cascade does not claim novelty for distillation, cascades, or threshold selection with guarantees. It builds on tinydistill (MIT), bounded-distill (Apache-2.0), FrugalGPT (arXiv:2305.05176), and Learn then Test (Angelopoulos, Bates, Candès, Jordan, Lei; arXiv:2110.01052, DOI 10.1214/24-AOAS1998). See `CITATION.cff`.
