import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime

import numpy as np
from PIL import Image, ImageSequence

try:
    import torch
except Exception:
    torch = None

try:
    import folder_paths
except Exception:
    folder_paths = None


def _safe_name(name: str) -> str:
    name = str(name or "alpha_mov").strip()
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:120] or "alpha_mov"


def _get_output_dir() -> str:
    if folder_paths is not None:
        return folder_paths.get_output_directory()
    return os.path.abspath(os.path.join(os.getcwd(), "output"))


def _unique_mov_path(filename_prefix: str) -> str:
    out_dir = os.path.join(_get_output_dir(), "alpha_mov")
    os.makedirs(out_dir, exist_ok=True)

    prefix = _safe_name(filename_prefix)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"{prefix}_{stamp}")
    path = base + ".mov"

    i = 1
    while os.path.exists(path):
        path = f"{base}_{i:03d}.mov"
        i += 1
    return path


def _resolve_ffmpeg(ffmpeg_path: str) -> str:
    ffmpeg_path = (ffmpeg_path or "ffmpeg").strip()
    found = shutil.which(ffmpeg_path)
    if found:
        return found
    if os.path.exists(ffmpeg_path):
        return ffmpeg_path
    raise RuntimeError(
        f"Cannot find ffmpeg: {ffmpeg_path}. "
        "Please install ffmpeg and make sure it is in PATH, or set ffmpeg_path."
    )


def _run_ffmpeg_png_sequence_to_mov(frame_dir: str, fps: float, output_path: str, codec: str, ffmpeg_path: str):
    ffmpeg = _resolve_ffmpeg(ffmpeg_path)
    fps = float(fps)
    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    input_pattern = os.path.join(frame_dir, "frame_%06d.png")

    # ProRes 4444: good for AE / Premiere / Final Cut. Keeps real alpha and partial transparency.
    if codec == "prores_4444":
        cmd = [
            ffmpeg, "-y",
            "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-vf", "format=rgba",
            "-c:v", "prores_ks",
            "-profile:v", "4444",
            "-pix_fmt", "yuva444p10le",
            "-alpha_bits", "16",
            "-vendor", "apl0",
            "-an",
            output_path,
        ]
    # PNG-in-MOV: larger file, but very direct/lossless-style alpha.
    elif codec == "png_mov":
        cmd = [
            ffmpeg, "-y",
            "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-vf", "format=rgba",
            "-c:v", "png",
            "-pix_fmt", "rgba",
            "-an",
            output_path,
        ]
    # Animation codec: huge, but some old tools like it.
    elif codec == "qtrle":
        cmd = [
            ffmpeg, "-y",
            "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-vf", "format=rgba",
            "-c:v", "qtrle",
            "-pix_fmt", "argb",
            "-an",
            output_path,
        ]
    else:
        raise ValueError(f"Unsupported codec: {codec}")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr}"
        )


def _image_tensor_to_rgba_uint8(images):
    """
    ComfyUI IMAGE usually is BHWC float tensor in range 0..1.
    This also supports HWC arrays and RGBA images from JoinImageWithAlpha.
    """
    if torch is not None and isinstance(images, torch.Tensor):
        arr = images.detach().cpu().numpy()
    else:
        arr = np.asarray(images)

    if arr.ndim == 3:
        arr = arr[None, ...]

    if arr.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor with shape BHWC or HWC, got shape {arr.shape}")

    if arr.shape[-1] == 1:
        rgb = np.repeat(arr[..., :1], 3, axis=-1)
        alpha = np.ones_like(arr[..., :1])
        arr = np.concatenate([rgb, alpha], axis=-1)
    elif arr.shape[-1] == 3:
        alpha = np.ones(arr.shape[:-1] + (1,), dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=-1)
    elif arr.shape[-1] >= 4:
        arr = arr[..., :4]
    else:
        raise ValueError(f"Unsupported channel count: {arr.shape[-1]}")

    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _write_rgba_frames_from_tensor(images, frame_dir: str):
    arr = _image_tensor_to_rgba_uint8(images)
    for i, frame in enumerate(arr):
        Image.fromarray(frame, mode="RGBA").save(os.path.join(frame_dir, f"frame_{i:06d}.png"))
    return int(arr.shape[0])


def _write_rgba_frames_from_webp(webp_path: str, frame_dir: str):
    if not os.path.exists(webp_path):
        raise FileNotFoundError(f"WebP file not found: {webp_path}")

    im = Image.open(webp_path)
    count = 0
    for frame in ImageSequence.Iterator(im):
        rgba = frame.convert("RGBA")
        rgba.save(os.path.join(frame_dir, f"frame_{count:06d}.png"))
        count += 1

    if count <= 0:
        raise RuntimeError(f"No frames decoded from WebP: {webp_path}")
    return count


class SaveMOVWithAlpha:
    """
    Save ComfyUI IMAGE frames to .mov with an alpha channel.
    Best connected directly after JoinImageWithAlpha.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "AI_Studio_alpha", "multiline": False}),
                "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "codec": (["prores_4444", "png_mov", "qtrle"], {"default": "prores_4444"}),
            },
            "optional": {
                "ffmpeg_path": ("STRING", {"default": "ffmpeg", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("mov_path", "frame_count")
    FUNCTION = "save_mov"
    CATEGORY = "AI Studio/Video"
    OUTPUT_NODE = True

    def save_mov(self, images, filename_prefix, fps, codec="prores_4444", ffmpeg_path="ffmpeg"):
        output_path = _unique_mov_path(filename_prefix)

        with tempfile.TemporaryDirectory(prefix="comfy_alpha_mov_") as tmp:
            frame_count = _write_rgba_frames_from_tensor(images, tmp)
            _run_ffmpeg_png_sequence_to_mov(tmp, fps, output_path, codec, ffmpeg_path)

        return (output_path, frame_count)


class WebPToMOVWithAlpha:
    """
    Convert animated/static WebP to .mov with an alpha channel.
    Use this for already-saved WebP files.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "webp_path": ("STRING", {"default": "", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "AI_Studio_from_webp", "multiline": False}),
                "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "codec": (["prores_4444", "png_mov", "qtrle"], {"default": "prores_4444"}),
            },
            "optional": {
                "ffmpeg_path": ("STRING", {"default": "ffmpeg", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("mov_path", "frame_count")
    FUNCTION = "convert"
    CATEGORY = "AI Studio/Video"
    OUTPUT_NODE = True

    def convert(self, webp_path, filename_prefix, fps, codec="prores_4444", ffmpeg_path="ffmpeg"):
        # Convenience: allow relative paths under ComfyUI/output.
        if not os.path.isabs(webp_path):
            candidate = os.path.join(_get_output_dir(), webp_path)
            if os.path.exists(candidate):
                webp_path = candidate

        output_path = _unique_mov_path(filename_prefix)

        with tempfile.TemporaryDirectory(prefix="comfy_webp_to_mov_") as tmp:
            frame_count = _write_rgba_frames_from_webp(webp_path, tmp)
            _run_ffmpeg_png_sequence_to_mov(tmp, fps, output_path, codec, ffmpeg_path)

        return (output_path, frame_count)


NODE_CLASS_MAPPINGS = {
    "SaveMOVWithAlpha": SaveMOVWithAlpha,
    "WebPToMOVWithAlpha": WebPToMOVWithAlpha,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveMOVWithAlpha": "Save MOV With Alpha",
    "WebPToMOVWithAlpha": "WebP To MOV With Alpha",
}
