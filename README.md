# ComfyUI-WebPAlphaMOV

两个 ComfyUI 自定义节点：

1. **Save MOV With Alpha**  
   直接接 ComfyUI 的 `IMAGE` / RGBA 帧，保存为带 Alpha 通道的 `.mov`。  
   推荐接在 `JoinImageWithAlpha` 后面。

2. **WebP To MOV With Alpha**  
   把已有的 animated/static `.webp` 转成带 Alpha 通道的 `.mov`。

## 安装

把整个 `ComfyUI-WebPAlphaMOV` 文件夹放到：

```bash
ComfyUI/custom_nodes/
```

安装依赖：

```bash
cd ComfyUI
pip install -r custom_nodes/ComfyUI-WebPAlphaMOV/requirements.txt
```

安装 FFmpeg：

macOS:

```bash
brew install ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

Windows:
下载 FFmpeg 后，把 `ffmpeg.exe` 所在目录加入 PATH，或者在节点的 `ffmpeg_path` 填完整路径。

## 推荐接法

你当前的 Alpha workflow 推荐：

```text
RGB VAE Decode
Alpha VAE Decode
        ↓
ImageToMask / InvertMask
        ↓
JoinImageWithAlpha
        ↓
Save MOV With Alpha
```

也就是说，不一定要先保存 WebP 再转 MOV。直接从 `JoinImageWithAlpha` 输出 MOV，透明边缘和半透明对象会更干净。

## Codec 选择

- `prores_4444`：推荐，适合 AE / Premiere / Final Cut。
- `png_mov`：文件更大，但透明通道直接。
- `qtrle`：兼容一些老工具，文件很大。

输出路径：

```text
ComfyUI/output/alpha_mov/
```
