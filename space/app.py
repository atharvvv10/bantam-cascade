"""Hugging Face Space entry point for Bantam Cascade.

Set ``BANTAM_DEMO_RUN_DIR`` to a mounted run directory containing
``cascade_results.jsonl`` and ``students/<name>.joblib``. No provider key is
used; the Space shows student-only decisions at frozen thresholds.
"""

from bantam_cascade.gradio_app import main

if __name__ == "__main__":
    main()
