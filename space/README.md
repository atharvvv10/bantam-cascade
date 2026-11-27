# Bantam Cascade demo Space

This directory is ready to copy into a Hugging Face Space with the Gradio SDK.

- `app.py` starts the demo.
- `Dockerfile` uses Python 3.11 and installs `git` for the source install.
- `requirements.txt` installs the package from the repository's `main` branch.
  Pin this to a released tag before a public Space launch.
- Set `BANTAM_DEMO_RUN_DIR` to a mounted directory containing
  `cascade_results.jsonl` and `students/<name>.joblib`. Those run artifacts are
  git-ignored in the source repository and are **not** published here.

The Space shows student-only decisions at the frozen calibration thresholds. It
has no provider key, so it does not call a teacher, does not fall back, and does
not produce a certificate. The local terminal GIF in `docs/assets/` is a
separate illustration of the CLI evidence, not a Space screenshot.
