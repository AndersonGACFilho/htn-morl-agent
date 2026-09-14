"""Construction of the recording console used to draw and export frames.

The panels use box-drawing and arrow glyphs. A Windows console defaults to a
legacy code page that cannot encode them, so the output stream is switched to
UTF-8 and the legacy renderer is disabled before anything is drawn.
"""

from __future__ import annotations

import sys

from rich.console import Console

CONSOLE_WIDTH = 104
CONSOLE_HEIGHT = 40


def build_console(
    *, width: int = CONSOLE_WIDTH, height: int = CONSOLE_HEIGHT
) -> Console:
    """Return a recording console able to emit the glyphs the panels use.

    Args:
        width: Console width in characters.
        height: Console height in lines.

    Returns:
        A console that records its output for later SVG export.
    """
    _enable_utf8_output()

    return Console(record=True, width=width, height=height, legacy_windows=False)


def _enable_utf8_output() -> None:
    """Switch standard output to UTF-8 when the stream supports it."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)

    if reconfigure is None:
        return

    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        return
