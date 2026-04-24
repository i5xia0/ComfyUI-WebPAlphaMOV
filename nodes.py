import os
import re
import shutil
import subprocess
import tempfile

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


def _allocate_output_path(filename_prefix: str, width: int, height: int) -> tuple[str, str, str]:
    """
    遵循 ComfyUI 官方 SaveImage 命名约定：使用 folder_paths.get_save_image_path 生成
    形如 `{prefix}_00001_.mov` 的递增文件名。返回 (绝对路径, subfolder, filename)。

    filename_prefix 允许包含子目录，比如 "alpha_mov/AI Studio"，会自动创建子目录。
    """
    prefix = _safe_name(filename_prefix)

    if folder_paths is None:
        # 未在 ComfyUI 运行时，退化到手动自增命名
        out_dir = _get_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        i = 1
        while True:
            filename = f"{prefix}_{i:05d}_.mov"
            path = os.path.join(out_dir, filename)
            if not os.path.exists(path):
                return path, "", filename
            i += 1

    # ComfyUI 标准：get_save_image_path 会处理 prefix 带子目录、自增 counter、清洗路径
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, _get_output_dir(), width, height,
    )
    os.makedirs(full_output_folder, exist_ok=True)
    # counter 递增直到找到不存在的文件名，和官方 SaveImage 行为一致
    while True:
        file_name = f"{filename}_{counter:05}_.mov"
        file_path = os.path.join(full_output_folder, file_name)
        if not os.path.exists(file_path):
            break
        counter += 1
    return file_path, subfolder, file_name


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
        # 用第一帧的尺寸填 get_save_image_path 的 width/height 参数（它只用于 prefix 替换）
        if torch is not None and isinstance(images, torch.Tensor):
            _, h, w = images.shape[0], images.shape[1], images.shape[2]
        else:
            arr = np.asarray(images)
            if arr.ndim == 3:
                h, w = arr.shape[0], arr.shape[1]
            else:
                h, w = arr.shape[1], arr.shape[2]

        output_path, subfolder, filename = _allocate_output_path(filename_prefix, int(w), int(h))

        with tempfile.TemporaryDirectory(prefix="comfy_alpha_mov_") as tmp:
            frame_count = _write_rgba_frames_from_tensor(images, tmp)
            _run_ffmpeg_png_sequence_to_mov(tmp, fps, output_path, codec, ffmpeg_path)

        return {
            "ui": {
                "gifs": [{
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": "output",
                    "format": "video/quicktime",
                    "frame_rate": float(fps),
                }]
            },
            "result": (output_path, frame_count),
        }


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

        # 读一帧拿尺寸给 get_save_image_path；失败就用占位
        try:
            with Image.open(webp_path) as probe:
                w, h = probe.size
        except Exception:  # noqa: BLE001
            w, h = 512, 512

        output_path, subfolder, filename = _allocate_output_path(filename_prefix, int(w), int(h))

        with tempfile.TemporaryDirectory(prefix="comfy_webp_to_mov_") as tmp:
            frame_count = _write_rgba_frames_from_webp(webp_path, tmp)
            _run_ffmpeg_png_sequence_to_mov(tmp, fps, output_path, codec, ffmpeg_path)

        return {
            "ui": {
                "gifs": [{
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": "output",
                    "format": "video/quicktime",
                    "frame_rate": float(fps),
                }]
            },
            "result": (output_path, frame_count),
        }


NODE_CLASS_MAPPINGS = {
    "SaveMOVWithAlpha": SaveMOVWithAlpha,
    "WebPToMOVWithAlpha": WebPToMOVWithAlpha,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveMOVWithAlpha": "Save MOV With Alpha",
    "WebPToMOVWithAlpha": "WebP To MOV With Alpha",
}
