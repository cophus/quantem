"""Tilt-series movies: panels side by side, drawn with show_2d, encoded to H.264 by ffmpeg."""

import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from quantem.core.visualization.visualization import show_2d


def save_movie(panels, titles, path, vmin, vmax, fps=12, axsize=(4, 4), dpi=100):
    """panels: list of arrays (n_frames, h, w), drawn left to right with one fixed gray scale
    vmin..vmax. titles: function frame -> list of panel titles. Writes an mp4 to path."""
    n = len(panels[0])
    norm = {"interval_type": "manual", "vmin": vmin, "vmax": vmax}
    ffmpeg = shutil.which("ffmpeg")
    proc = None
    for t in range(n):
        fig, _ = show_2d(
            [p[t] for p in panels],
            norm=norm,
            cmap="gray",
            title=titles(t),
            axsize=axsize,
        )
        fig.set_dpi(dpi)
        fig.canvas.draw()
        rgb = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
        if proc is None:
            h, w = rgb.shape[:2]
            cmd = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        proc.stdin.write(rgb.tobytes())
    proc.stdin.close()
    proc.wait()
    return path
