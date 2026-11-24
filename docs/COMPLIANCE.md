# Compliance checklist

> Status: incomplete. Nothing may be released until every item is checked.

- [ ] Read Groq's current terms and the chosen model's license.
- [ ] Confirm that training a smaller model on teacher outputs and publishing the model (and possibly the labels) is allowed. If unclear: do NOT publish labels or weights; say so here.
- [ ] Check licenses of `PolyAI/banking77` (CC BY 4.0, verified via HF API 2026-11-24) and `clinc/clinc_oos` (CC BY 3.0, verified via HF API 2026-11-24) before redistributing anything.
- [ ] Check licenses of the embedding models (`model2vec`, MiniLM) before use or redistribution.
- [ ] Check licenses of every dependency before redistributing anything.
- [ ] Credit prior work in README, `CITATION.cff`, and `docs/METHODOLOGY.md`.
- [ ] Confirm no API keys or secrets exist in the repository history.
- [ ] Confirm `bantam-cascade` is still free on PyPI immediately before the first release.
