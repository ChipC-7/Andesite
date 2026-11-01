"""视觉请求的图片预处理。

大截图会占用大量视觉 token，并拖慢首次推理。发送前把长边压到 1280、
转成 JPEG；用户需要原图时可在设置里关闭压缩。
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_IMAGE_EDGE = 1280  # 最长边上限
JPEG_QUALITY = 88  # 压缩质量
RECOMPRESS_THRESHOLD_BYTES = 1_200_000  # 超过 1.2 MB 才重新压缩


def compress_image(
    data: bytes,
    name: str,
    mime: str,
    *,
    max_edge: int = MAX_IMAGE_EDGE,
    quality: int = JPEG_QUALITY,
) -> tuple[bytes, str, str]:
    """返回 ``(数据, MIME, 文件名)``；不需要压缩时原样返回。"""
    if not data:
        return data, mime, name
    # Pillow 只在第一次压缩图片时导入，避免拖慢应用启动。
    try:
        from PIL import Image, ImageOps
    except ImportError:  # pragma: no cover - 仅在未安装 Pillow 时触发
        return data, mime, name
    try:
        with Image.open(io.BytesIO(data)) as source:
            # 先根据 EXIF 方向摆正图片，避免手机照片旋转。
            source = ImageOps.exif_transpose(source)
            if getattr(source, "is_animated", False):
                source.seek(0)  # 动图只保留首帧
            width, height = source.size
            needs_resize = max(width, height) > max_edge
            needs_recompress = len(data) > RECOMPRESS_THRESHOLD_BYTES
            if not needs_resize and not needs_recompress:
                return data, mime, name

            if needs_resize:
                source.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

            # JPEG 不支持透明通道，透明图先铺白底。
            if source.mode in {"RGBA", "LA"} or (
                source.mode == "P" and "transparency" in source.info
            ):
                rgba = source.convert("RGBA")
                flattened = Image.new("RGB", rgba.size, (255, 255, 255))
                flattened.paste(rgba, mask=rgba.getchannel("A"))
                prepared = flattened
            else:
                prepared = source.convert("RGB")

            buffer = io.BytesIO()
            prepared.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
            compressed = buffer.getvalue()
            # 重压缩反而变大时，继续用原图。
            if not needs_resize and len(compressed) >= len(data):
                return data, mime, name
            return compressed, "image/jpeg", f"{Path(name).stem}.jpg"
    except Exception:  # noqa: BLE001 - 坏图不能阻止添加附件
        logger.warning("Could not compress image %s", name, exc_info=True)
        return data, mime, name
