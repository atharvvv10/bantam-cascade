# Bantam Cascade demo Space

This directory is a self-contained Gradio Space.

- `app.py` starts the demo. If `bantam_cascade` is not installed, it clones the
  pinned `v0.1.1` source tag at startup and imports from `src/`. This avoids
  pip-resolving the released package's pinned `pydantic==2.14.4`, which
  conflicts with the HF Gradio SDK's `pydantic<=2.12.5`.
- `requirements.txt` lists only the NumPy/SciPy/scikit-learn/joblib packages
  the saved `word_tfidf` student needs.
- `runs/sms-groq/` contains the frozen `word_tfidf` threshold row and the
  gold-label-trained student artifact used by the deployed Space; `run/` is the
  local copy for the same scaffold. No teacher labels, teacher caches, provider
  keys, or certificates are included.
- The Space shows student-only decisions at frozen calibration thresholds. It
  has no provider key, so it does not call a teacher, does not fall back, and
  does not produce a certificate. The local terminal GIF in `docs/assets/` is a
  separate illustration of the CLI evidence, not a Space screenshot.
