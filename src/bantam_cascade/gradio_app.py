"""Optional Gradio demo for saved Bantam Cascade students.

The Space has no provider key, so this app shows student-only decisions with the
frozen calibration thresholds. It never claims a new certificate and never calls
a teacher API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, cast

from bantam_cascade.students import restore_student
from bantam_cascade.students.base import confidence_scores, labels_from_proba


class Launchable(Protocol):
    """Minimal interface needed to launch a Gradio app."""

    def launch(self, *, server_name: str, server_port: int) -> None:
        """Start the web server."""
        ...


def read_thresholds(run_dir: Path) -> dict[str, float]:
    """Read the frozen calibration thresholds from cascade result rows."""
    path = run_dir / "cascade_results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    thresholds: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        thresholds[str(row["student"])] = float(row["threshold"])
    if not thresholds:
        raise ValueError(f"No cascade results in {path}")
    return thresholds


def student_report(run_dir: Path, text: str) -> str:
    """Return a Markdown report of student-only decisions for one text."""
    thresholds = read_thresholds(run_dir)
    students_dir = run_dir / "students"
    lines = [
        "**Offline student-only demo** — no teacher fallback and no new certificate.",
        "",
        f"Input: {text!r}",
        "",
        "| Student | Frozen threshold | Label | Confidence | Fallback needed |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for name, threshold in sorted(thresholds.items()):
        path = students_dir / f"{name}.joblib"
        if not path.is_file():
            lines.append(f"| {name} | {threshold:.4f} | artifact missing | — | — |")
            continue
        student = restore_student(path)
        proba = student.predict_proba([text])
        label = labels_from_proba(proba, student.classes)[0]
        confidence = float(confidence_scores(proba)[0])
        fallback = "yes" if confidence < threshold else "no"
        lines.append(f"| {name} | {threshold:.4f} | {label} | {confidence:.4f} | {fallback} |")
    lines.extend(
        [
            "",
            "When fallback is needed, the deployed cascade calls the pinned teacher; "
            "that call is disabled here because the demo has no API key.",
        ]
    )
    return "\n".join(lines)


def build_interface(run_dir: Path) -> Launchable:
    """Build a Gradio interface; gradio is imported only when this is called."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the demo extra: pip install -e '.[demo]'") from exc

    def predict(text: str) -> str:
        return student_report(run_dir, text)

    interface = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(label="Text to classify", lines=3),
        outputs=gr.Markdown(label="Student-only cascade view"),
        title="Bantam Cascade — offline student demo",
        description=(
            "Frozen calibration thresholds from a saved run. No teacher API calls, "
            "no certification run, and no deployment guarantee."
        ),
    )
    return cast(Launchable, interface)


def main() -> None:
    """Launch the Gradio app using ``BANTAM_DEMO_RUN_DIR``."""
    raw = os.environ.get("BANTAM_DEMO_RUN_DIR", "runs/sms-groq")
    run_dir = Path(raw)
    interface = build_interface(run_dir)
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))


if __name__ == "__main__":
    main()
