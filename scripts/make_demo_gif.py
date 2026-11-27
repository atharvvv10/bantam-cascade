"""Render a small terminal-style GIF from real, frozen result numbers.

Run from the repository root:

    python scripts/make_demo_gif.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FRAMES: tuple[tuple[str, ...], ...] = (
    (
        "$ bantam-cascade certify --config configs/sms-spam-groq.yaml",
        "",
        "Status: CERTIFIED",
        "4 errors / 300 | p = 0.001878 | delta = 0.050",
        "0 fallback calls | p95 = 0.7 ms | prior looks = 0",
    ),
    (
        "$ bantam-cascade validity --trials 200 --examples 600",
        "",
        "boundary_below_target: valid false certs 0.000",
        "naive-threshold false certs 0.040 (allowed 0.050)",
        "above_target certification rate: valid 0.810 | naive 0.990",
    ),
    (
        "$ bantam-cascade plan --assumed-accuracy 0.98",
        "",
        "Target 0.950 | confidence 0.950",
        "Minimum certification examples: 240",
        "Original SMS run used: 300",
    ),
    (
        "$ bantam-cascade reproduce",
        "",
        "Frozen certificate verified: True",
        "Offline evidence: verification, plan, validity, statuses",
        "No certification split was re-labelled or tested again.",
    ),
)


def _font(size: int) -> ImageFont.ImageFont:
    """Load a monospace font when available, otherwise Pillow's default."""
    candidates = (
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def render() -> list[Image.Image]:
    """Render one image per terminal frame."""
    width, height = 1100, 360
    font = _font(22)
    frames: list[Image.Image] = []
    for lines in FRAMES:
        image = Image.new("RGB", (width, height), color=(17, 20, 26))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width - 1, 54), fill=(34, 39, 48))
        draw.text((22, 14), "bantam-cascade", font=font, fill=(126, 220, 168))
        for index, line in enumerate(lines):
            color = (235, 235, 235) if index else (126, 220, 168)
            draw.text((24, 84 + index * 44), line, font=font, fill=color)
        frames.append(image)
    return frames


def main() -> None:
    """Write the animated GIF under docs/assets."""
    frames = render()
    output = Path("docs/assets/bantam-cascade.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=[2600, 2600, 2600, 3200],
        loop=0,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
