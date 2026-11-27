# Bantam Cascade demo Space

This directory is a self-contained Gradio Space.

- `app.py` starts the demo.
- `requirements.txt` installs `bantam-cascade` from tag `v0.1.1` with
  `--no-deps` and lists only the NumPy/SciPy/scikit-learn/joblib packages the
  saved `word_tfidf` student needs. This avoids the HF Gradio SDK's
  `pydantic<=2.12.5` constraint without changing the released package.
- `run/` contains only the frozen `word_tfidf` threshold row and the
  gold-label-trained student artifact. No teacher labels, teacher caches,
  provider keys, or certificates are included.
- The Space shows student-only decisions at frozen calibration thresholds. It
  has no provider key, so it does not call a teacher, does not fall back, and
  does not produce a certificate. The local terminal GIF in `docs/assets/` is a
  separate illustration of the CLI evidence, not a Space screenshot.
