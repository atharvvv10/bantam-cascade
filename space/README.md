# Bantam Cascade demo Space

This directory is a self-contained Gradio Space.

- `app.py` starts the demo.
- `Dockerfile` uses Python 3.11, installs `git`, installs
  `bantam-cascade[students]` from tag `v0.1.1`, and sets
  `BANTAM_DEMO_RUN_DIR=/app/run`.
- `run/` contains only the frozen `word_tfidf` threshold row and the
  gold-label-trained student artifact. No teacher labels, teacher caches,
  provider keys, or certificates are included.
- The Space shows student-only decisions at frozen calibration thresholds. It
  has no provider key, so it does not call a teacher, does not fall back, and
  does not produce a certificate. The local terminal GIF in `docs/assets/` is a
  separate illustration of the CLI evidence, not a Space screenshot.
