"""Reading and validating video clips for MiniMax H3 training.

Fizgig does not PREPARE clips — Gizmo does. This module only decides whether a file is already
on spec, and if it is, decodes it. Nothing here transcodes, resamples or trims: the moment a
training app starts silently fixing footage, two datasets that look identical train differently
and nobody can tell why.

The spec (also in the README, so it can be hit by hand):

    container    .mp4
    frame rate   exactly 24 fps — H3's own clock
    frames       on the 17n+5 grid: 5, 22, 39
    size         multiples of 32
    audio        32 kHz stereo, or no track at all
    muting       a `_mute` suffix on the stem trains that clip's video only

Off-spec files are REFUSED with the value found and what to do about it, because the failure they
would otherwise cause is invisible: a 30 fps clip trains motion at the wrong speed and looks
perfectly fine doing it.
"""

import math
import os
import re
import subprocess

import numpy as np

from fizgig.minimax.model import FPS, pixel_frames_for_latent

VIDEO_EXTENSIONS = (".mp4",)
AUDIO_SAMPLE_RATE = 32000
AUDIO_CHANNELS = 2
SIZE_STEP = 32
MUTE_SUFFIX = "_mute"

# Every length the VAE can encode: 17n+5, which is 5, 22, 39, 56, 73, 90, 107, 124.
#
# The long end used to be withheld on cost grounds. That was the wrong place to make the
# decision: a 124-frame clip is unaffordable on a 16 GB card and perfectly reasonable on a 96 GB
# one, and refusing it here takes the choice away from the person who knows which they have.
# Gizmo shows what each length needs; this module's job is only to say what the model accepts.
GRID_FRAMES = tuple(pixel_frames_for_latent(5 * n + 2) for n in range(8))


class ClipRejected(ValueError):
    """A clip that does not meet the spec. The message is user-facing — it names the value found
    and what to do, because a silent trim or resample is how a dataset goes quietly wrong."""


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def is_muted(path: str) -> bool:
    """`walk_03_mute.mp4` -> True. The suffix wins over the file's contents: it is a statement of
    intent, and the audio stays in the file so the decision is reversible by rename."""
    return os.path.splitext(os.path.basename(path))[0].lower().endswith(MUTE_SUFFIX)


def _ffmpeg() -> str:
    from fizgig.lora_royale.export import _find_ffmpeg
    exe = _find_ffmpeg()
    if not exe:
        raise ClipRejected("no ffmpeg available — reinstall to restore the bundled imageio-ffmpeg")
    return exe


def probe(path: str) -> dict:
    """Container facts, read from ffmpeg's own stream banner. No decode.

    ffmpeg with no output exits non-zero and prints the streams to stderr; that is the documented
    way to get this without ffprobe, which imageio-ffmpeg does not bundle.
    """
    out = subprocess.run([_ffmpeg(), "-hide_banner", "-i", path],
                         capture_output=True, text=True).stderr
    info = {"fps": None, "width": None, "height": None,
            "has_audio": False, "sample_rate": None, "channels": None}

    m = re.search(r"Stream #\d+:\d+.*?: Video: .*?(\d{2,5})x(\d{2,5})", out, re.S)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s+fps", out)
    if m:
        info["fps"] = float(m.group(1))

    m = re.search(r"Stream #\d+:\d+.*?: Audio: [^,]+, (\d+) Hz, (\w+)", out)
    if m:
        info["has_audio"] = True
        info["sample_rate"] = int(m.group(1))
        info["channels"] = {"mono": 1, "stereo": 2}.get(m.group(2), None)
    if info["width"] is None or info["fps"] is None:
        raise ClipRejected(f"{os.path.basename(path)}: no readable video stream")
    return info


def validate(path: str, info: dict = None, frames: int = None) -> dict:
    """Raise ClipRejected unless the file is on spec. Returns the probe dict.

    frames, when known (the caller has just decoded), is checked against the grid. Frame COUNT is
    the one thing the banner cannot be trusted for — duration x fps rounds — so it is verified
    after decoding rather than guessed here.
    """
    info = info or probe(path)
    name = os.path.basename(path)
    where = "Prepare it with Gizmo, or see the clip spec in the README."

    if abs(info["fps"] - FPS) > 0.01:
        raise ClipRejected(
            f"{name}: {info['fps']:g} fps, but H3 runs at {FPS}. Training it as-is would learn "
            f"motion at the wrong speed and look fine doing it. {where}")

    for dim, label in ((info["width"], "width"), (info["height"], "height")):
        if dim % SIZE_STEP:
            raise ClipRejected(
                f"{name}: {info['width']}x{info['height']} — {label} must be a multiple of "
                f"{SIZE_STEP}. {where}")

    if info["has_audio"]:
        if info["sample_rate"] != AUDIO_SAMPLE_RATE:
            raise ClipRejected(
                f"{name}: audio is {info['sample_rate']} Hz, but the H3 audio VAE is "
                f"{AUDIO_SAMPLE_RATE} Hz. {where}")
        if info["channels"] not in (None, AUDIO_CHANNELS):
            raise ClipRejected(
                f"{name}: audio is {info['channels']}-channel, H3 packs stereo. {where}")

    if frames is not None and frames not in GRID_FRAMES:
        raise ClipRejected(
            f"{name}: {frames} frames. H3 encodes on a 17n+5 grid, so a clip has to be one of "
            f"{', '.join(str(f) for f in GRID_FRAMES)} frames. {where}")
    return info


def read_frames(path: str) -> np.ndarray:
    """Decode to uint8 (T, H, W, 3) RGB. Every frame — clips are 39 at most."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(path, pix_fmt="rgb24")
    meta = next(reader)
    w, h = meta["size"]
    out = [np.frombuffer(f, dtype=np.uint8).reshape(h, w, 3) for f in reader]
    if not out:
        raise ClipRejected(f"{os.path.basename(path)}: decoded no frames")
    return np.stack(out)


def read_audio(path: str) -> np.ndarray:
    """Decode to float32 (2, L) at 32 kHz in [-1, 1], or None when there is no track.

    Muted clips return None here too: the `_mute` suffix is checked FIRST, so the audio is never
    read at all and the item simply has no audio target. That is deliberately not the same as a
    target of silence, which would teach the model that this footage sounds like nothing.
    """
    if is_muted(path) or not probe(path)["has_audio"]:
        return None
    raw = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", path,
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_SAMPLE_RATE), "-"],
        capture_output=True).stdout
    if not raw:
        return None
    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1, AUDIO_CHANNELS).T.copy()
    # A track that decodes to digital silence counts as muted. Gizmo cannot produce one, but a
    # hand-prepped clip can, and a silence target is worse than no target at all.
    if not np.any(wav):
        return None
    return wav


def expected_audio_samples(frames: int) -> int:
    """Samples covering `frames` pixel frames — what a clip's audio should decode to."""
    return int(math.ceil(frames / FPS * AUDIO_SAMPLE_RATE))
