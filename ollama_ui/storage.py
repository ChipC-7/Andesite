"""会话、设置和聊天图片的本地持久化。"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import Attachment, Conversation, Settings

logger = logging.getLogger(__name__)

# MIME 类型对应的文件后缀。
_EXTENSIONS_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}
# 仅允许这些图片后缀，避免写入任意文件。
_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


class Store:
    """读写 ``~/.ollama-ui``（或 OLLAMA_UI_HOME 指定目录）下的应用数据。"""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        env_dir = os.environ.get("OLLAMA_UI_HOME")
        root = data_dir or env_dir or Path.home() / ".ollama-ui"
        self.data_dir = Path(root).expanduser()
        self.media_dir = self.data_dir / "media"
        self.conversations_path = self.data_dir / "conversations.json"
        self.settings_path = self.data_dir / "settings.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    def load_conversations(self) -> list[Conversation]:
        raw = self._read_json(self.conversations_path)
        if raw is None:
            return []
        # 兼容旧版直接存列表和当前版本带信封的格式。
        items: Any = raw.get("conversations") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            logger.warning("Unexpected conversations.json shape; ignoring it.")
            return []
        conversations: list[Conversation] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                conversation = Conversation.from_dict(item)
            except Exception:  # noqa: BLE001 - corrupt individual entry must not kill startup
                logger.exception("Skipping invalid conversation entry.")
                continue
            # 跳过重复 ID，防止侧栏出现两条相同会话。
            if conversation.id in seen:
                continue
            seen.add(conversation.id)
            conversations.append(conversation)
        conversations.sort(key=lambda c: c.last_message_at, reverse=True)
        logger.debug("Loaded %d conversations from %s", len(conversations), self.conversations_path)
        return conversations

    def save_conversations(self, conversations: Iterable[Conversation]) -> None:
        conversations = list(conversations)
        logger.debug("Saving %d conversations to %s", len(conversations), self.conversations_path)
        # 使用带版本号的信封，便于以后迁移数据格式。
        payload = {
            "version": 1,
            "conversations": [c.to_dict() for c in conversations],
        }
        self._write_json(self.conversations_path, payload)

    # ------------------------------------------------------------------
    # 设置
    # ------------------------------------------------------------------
    def load_settings(self) -> Settings:
        raw = self._read_json(self.settings_path)
        if isinstance(raw, dict):
            return Settings.from_dict(raw)
        return Settings()

    def save_settings(self, settings: Settings) -> None:
        self._write_json(self.settings_path, settings.to_dict())

    # ------------------------------------------------------------------
    # 图片
    # ------------------------------------------------------------------
    def save_image(self, name: str, mime: str, data: bytes) -> Attachment:
        # 文件名用 UUID，避免重名或路径注入；原始名只用于显示。
        extension = self._extension_for(name, mime)
        filename = f"{uuid.uuid4().hex}{extension}"
        target = self.media_dir / filename
        target.write_bytes(data)
        return Attachment(
            name=name or f"image{extension}",
            mime=mime or "image/png",
            path=filename,
            size=len(data),
        )

    def read_image(self, attachment: Attachment) -> bytes:
        # 相对路径一律从 media 目录解析。
        path = Path(attachment.path)
        if not path.is_absolute():
            path = self.media_dir / path
        return path.read_bytes()

    def delete_images(self, attachments: Iterable[Attachment]) -> None:
        for attachment in attachments:
            path = Path(attachment.path)
            if not path.is_absolute():
                path = self.media_dir / path
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove image %s", path, exc_info=True)

    def delete_conversation_images(self, conversation: Conversation) -> None:
        attachments = [
            attachment
            for message in conversation.messages
            for attachment in message.attachments
        ]
        self.delete_images(attachments)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    def _extension_for(name: str, mime: str) -> str:
        # 优先用原后缀，其次按 MIME 推断，未知类型回退 png。
        extension = Path(name).suffix.lower()
        if extension in _ALLOWED_EXTENSIONS:
            return ".jpg" if extension == ".jpeg" else extension
        return _EXTENSIONS_BY_MIME.get((mime or "").lower(), ".png")

    def _read_json(self, path: Path) -> Any:
        # 文件不存在或内容损坏时都返回 None，由调用方使用默认值。
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not read %s", path)
            return None

    def _write_json(self, path: Path, payload: Any) -> None:
        # 先写临时文件并 fsync，再原子替换，避免断电留下半个 JSON。
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: str | None = None
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=f"{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
