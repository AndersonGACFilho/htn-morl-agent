"""Persistence of rendered frames as SVG files and an animated GIF."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Sequence

import resvg_py
from PIL import Image
from rich.console import Console

from htn._examples.grid_world.core.theme import GridWorldTheme

DEFAULT_FRAME_DURATION_MS = 700


class SvgFrameExporter:
    """Write each rendered frame to its own SVG file.

    The exporter owns the output directory and the frame numbering, so the
    renderer stays free of file-system concerns and of a tick counter that used
    to drift from the simulation's own.
    """

    def __init__(
        self,
        output_directory: Path,
        theme: GridWorldTheme,
        title_prefix: str,
    ) -> None:
        """Initialize the exporter and create its output directory."""
        self._output_directory = output_directory
        self._theme = theme
        self._title_prefix = title_prefix
        self._output_directory.mkdir(parents=True, exist_ok=True)

    def export(self, console: Console, tick: int) -> Path:
        """Write the console's current contents as an SVG frame.

        Args:
            console: Recording console holding the frame.
            tick: Simulation tick used in the file name and title.

        Returns:
            The path of the written file.
        """
        title = f"{self._title_prefix} tick {tick:03d}"
        path = self._output_directory / f"{title}.svg"

        console.save_svg(str(path), title=title, theme=self._theme.terminal_theme())

        return path


class GifBuilder:
    """Assemble exported SVG frames into an animated GIF."""

    def __init__(self, frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS) -> None:
        """Initialize the builder.

        Raises:
            ValueError: If the frame duration is not positive.
        """
        if frame_duration_ms <= 0:
            raise ValueError("Frame duration must be greater than zero.")

        self._frame_duration_ms = frame_duration_ms

    def build(
        self,
        frames: Sequence[Path],
        output_path: Path,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Rasterize the frames and write an infinitely looping GIF.

        Args:
            frames: Paths of the SVG frames, in order.
            output_path: Destination of the GIF.
            progress: Optional callback receiving ``(index, total)``.

        Returns:
            The path of the written GIF.

        Raises:
            ValueError: If no frame is supplied.
        """
        if not frames:
            raise ValueError("At least one frame is required to build a GIF.")

        images = self._rasterize(frames, progress)
        canvas_size = self._canvas_size(images)
        padded = [self._pad(image, canvas_size) for image in images]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        padded[0].save(
            output_path,
            format="GIF",
            append_images=padded[1:],
            save_all=True,
            duration=self._frame_duration_ms,
            loop=0,
        )

        return output_path

    def _rasterize(
        self,
        frames: Sequence[Path],
        progress: Callable[[int, int], None] | None,
    ) -> list[Image.Image]:
        """Convert every SVG frame into an RGBA image."""
        images: list[Image.Image] = []
        total = len(frames)

        for index, frame in enumerate(frames, start=1):
            if progress is not None:
                progress(index, total)

            svg = frame.read_text(encoding="utf-8")
            png_bytes = bytes(resvg_py.svg_to_bytes(svg_string=svg))
            images.append(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))

        return images

    def _canvas_size(self, images: Sequence[Image.Image]) -> tuple[int, int]:
        """Return the smallest canvas holding every frame."""
        return (
            max(image.width for image in images),
            max(image.height for image in images),
        )

    def _pad(self, image: Image.Image, size: tuple[int, int]) -> Image.Image:
        """Center a frame on a canvas of the given size."""
        if image.size == size:
            return image

        canvas = Image.new("RGBA", size, (255, 255, 255, 255))
        offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
        canvas.paste(image, offset, image)

        return canvas
