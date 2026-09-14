"""Color palettes mirroring the LaTeX figures of the research manuscripts.

Keeping the rendered episode and the manuscript figures on one palette means a
screenshot of a run sits beside a TikZ diagram without a visual seam. Two
palettes exist because the dissertation and the ERAMIA article define their own
colors; both are reproduced here verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Palette:
    """Named colors shared by the terminal output, the SVG frames, and the GIF.

    Attributes:
        name: Identifier used to select the palette from the command line.
        htn_blue: Symbolic planning elements.
        morl_red: Learning and risk elements.
        game_green: Execution and success elements.
        fallback_orange: Recovery, cost, and warning elements.
        env_gray: Environment and muted elements.
        panel_background: Background of panels and empty tiles.
        panel_line: Panel borders and grid lines.
        ink_text: Default foreground text.
    """

    name: str
    htn_blue: str
    morl_red: str
    game_green: str
    fallback_orange: str
    env_gray: str
    panel_background: str
    panel_line: str
    ink_text: str


DISSERTATION_PALETTE = Palette(
    name="dissertation",
    htn_blue="#3465a4",
    morl_red="#b4463c",
    game_green="#48805c",
    fallback_orange="#c4782d",
    env_gray="#646464",
    panel_background="#f5f5f5",
    panel_line="#c8c8c8",
    ink_text="#1e1e1e",
)

ERAMIA_PALETTE = Palette(
    name="eramia",
    htn_blue="#2563eb",
    morl_red="#e11d48",
    game_green="#059669",
    fallback_orange="#d97706",
    env_gray="#64748b",
    panel_background="#f8fafc",
    panel_line="#e2e8f0",
    ink_text="#334155",
)

PALETTES: dict[str, Palette] = {
    DISSERTATION_PALETTE.name: DISSERTATION_PALETTE,
    ERAMIA_PALETTE.name: ERAMIA_PALETTE,
}


def get_palette(name: str) -> Palette:
    """Return a palette by name.

    Raises:
        ValueError: If no palette carries that name.
    """
    if name not in PALETTES:
        available = ", ".join(sorted(PALETTES))
        raise ValueError(f"Unknown palette {name!r}. Available: {available}")

    return PALETTES[name]
