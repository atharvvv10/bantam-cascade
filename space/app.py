"""Hugging Face Space entry point for Bantam Cascade.

Set ``BANTAM_DEMO_RUN_DIR`` to a mounted run directory containing
``cascade_results.jsonl`` and ``students/<name>.joblib``. No provider key is
used; the Space shows student-only decisions at frozen thresholds.

When the package is not installed, the pinned source tag is cloned at startup
and added to ``sys.path``. This keeps the Space free of the released package's
pinned dependencies, which conflict with the HF Gradio SDK. The unused
``@spaces.GPU`` marker satisfies HF's ZeroGPU startup check without changing
the CPU-only inference path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import spaces

_REPO_URL = "https://github.com/atharvvv10/bantam-cascade.git"
_REF = "v0.1.1"


@spaces.GPU
def _zero_gpu_startup_marker() -> str:
    """Unused marker required by HF ZeroGPU; the demo itself is CPU-only."""
    return "cpu-only"


def _ensure_package() -> None:
    """Import the package, cloning the pinned tag only when needed."""
    try:
        import bantam_cascade  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    target = Path(os.environ.get("BANTAM_SOURCE_DIR", "/tmp/bantam-cascade-src"))
    src = target / "src"
    if not (src / "bantam_cascade" / "gradio_app.py").is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", _REF, _REPO_URL, str(target)],
            check=True,
        )
    sys.path.insert(0, str(src))


_ensure_package()
from bantam_cascade.gradio_app import main  # noqa: E402

if __name__ == "__main__":
    main()
