"""会话、消息、图片和设置的数据模型。

所有对象都可序列化，用于本地保存和下次启动恢复。图片像素不写入 JSON，
附件只保存 media 目录里的文件名。
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def new_id() -> str:
    # 用短 UUID 作为消息/会话 ID。
    return uuid.uuid4().hex


def _as_float(value: Any, default: float = 0.0) -> float:
    # 宽松解析旧数据里的数字，失败时用默认值。
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _collapse(text: str) -> str:
    # 把连续空白压成一个空格，用于标题和预览。
    return re.sub(r"\s+", " ", text or "").strip()


def _clip(text: str, limit: int) -> str:
    # 超长文本截断并加省略号。
    text = _collapse(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


@dataclass
class Attachment:
    """已保存到 media 目录的图片附件。"""

    name: str  # 原始文件名，用于显示
    mime: str
    path: str  # media 目录内的相对路径
    size: int = 0
    id: str = field(default_factory=new_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "mime": self.mime,
            "path": self.path,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Attachment:
        return cls(
            id=str(raw.get("id") or new_id()),
            name=str(raw.get("name") or "image"),
            mime=str(raw.get("mime") or "image/png"),
            path=str(raw.get("path") or ""),
            size=_as_int(raw.get("size")),
        )


@dataclass
class PendingImage:
    """已选择/粘贴但还没发送的图片，只存在内存中。"""

    name: str
    mime: str
    data: bytes
    id: str = field(default_factory=new_id)


@dataclass
class SearchSource:
    """联网搜索返回的一条来源。"""

    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SearchSource:
        return cls(
            title=str(raw.get("title") or "网页"),
            url=str(raw.get("url") or ""),
            snippet=str(raw.get("snippet") or ""),
        )


@dataclass
class Message:
    """一条用户或助手消息。"""

    role: str
    content: str = ""
    attachments: list[Attachment] = field(default_factory=list)
    model: str | None = None
    thinking: str = ""  # 深度思考内容
    sources: list[SearchSource] = field(default_factory=list)  # 联网搜索来源
    error: bool = False
    # Ollama 返回的生成统计，用于显示 token 数和速度。
    eval_count: int = 0
    eval_duration_ns: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=new_id)

    @property
    def has_body(self) -> bool:
        # 空助手占位消息在保存时要被丢弃。
        return bool(self.content.strip() or self.thinking.strip() or self.attachments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "attachments": [a.to_dict() for a in self.attachments],
            "model": self.model,
            "thinking": self.thinking,
            "sources": [source.to_dict() for source in self.sources],
            "error": self.error,
            "eval_count": self.eval_count,
            "eval_duration_ns": self.eval_duration_ns,
            "prompt_eval_count": self.prompt_eval_count,
            "prompt_eval_duration_ns": self.prompt_eval_duration_ns,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Message:
        # 未知角色一律按用户消息处理，避免旧数据导致渲染异常。
        role = str(raw.get("role") or "user")
        if role not in {"user", "assistant", "system"}:
            role = "user"
        attachments_raw = raw.get("attachments") or []
        attachments = []
        if isinstance(attachments_raw, list):
            for item in attachments_raw:
                if isinstance(item, dict):
                    attachments.append(Attachment.from_dict(item))
        sources_raw = raw.get("sources") or []
        sources = []
        if isinstance(sources_raw, list):
            for item in sources_raw:
                if isinstance(item, dict) and item.get("url"):
                    sources.append(SearchSource.from_dict(item))
        return cls(
            id=str(raw.get("id") or new_id()),
            role=role,
            content=str(raw.get("content") or ""),
            attachments=attachments,
            model=(str(raw["model"]) if raw.get("model") else None),
            thinking=str(raw.get("thinking") or ""),
            sources=sources,
            error=bool(raw.get("error")),
            eval_count=_as_int(raw.get("eval_count")),
            eval_duration_ns=_as_int(raw.get("eval_duration_ns")),
            prompt_eval_count=_as_int(raw.get("prompt_eval_count")),
            prompt_eval_duration_ns=_as_int(raw.get("prompt_eval_duration_ns")),
            created_at=_as_float(raw.get("created_at"), time.time()),
        )


@dataclass
class Conversation:
    """一段完整对话。"""

    title: str = "新对话"
    model: str = ""
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    pinned: bool = False  # 置顶后排在侧栏最前
    id: str = field(default_factory=new_id)

    @property
    def preview(self) -> str:
        # 侧栏预览：取最后一条有内容的文本或图片消息。
        for message in reversed(self.messages):
            if message.role != "assistant" and message.role != "user":
                continue
            if message.error:
                return "生成失败"
            if message.content.strip():
                return _clip(message.content, 56)
            if message.attachments:
                return "图片对话"
        return "开始一段新的对话"

    @property
    def last_message_at(self) -> float:
        # 用于侧栏按最近活动排序。
        if self.messages:
            return max(self.updated_at, self.messages[-1].created_at)
        return self.updated_at

    def touch(self) -> None:
        self.updated_at = time.time()

    def matches(self, query: str) -> bool:
        """按标题、模型、正文、思考和图片名做不区分大小写的搜索。"""
        needle = (query or "").strip().casefold()
        if not needle:
            return True
        fields = [self.title, self.model]
        for message in self.messages:
            fields.extend((message.content, message.thinking))
            fields.extend(attachment.name for attachment in message.attachments)
        return any(needle in (value or "").casefold() for value in fields)

    def to_markdown(self) -> str:
        """导出为可移植的 Markdown 文档。"""
        created = datetime.fromtimestamp(self.created_at).strftime("%Y-%m-%d %H:%M")
        updated = datetime.fromtimestamp(self.updated_at).strftime("%Y-%m-%d %H:%M")
        lines = [f"# {self.title}", ""]
        if self.model:
            lines.append(f"- 模型：`{self.model}`")
        lines.append(f"- 创建：{created}")
        lines.append(f"- 更新：{updated}")
        lines.append("")

        for message in self.messages:
            if message.role == "user":
                lines.extend(["## 你", ""])
                lines.append(message.content.strip() or "_（仅发送了图片）_")
                if message.attachments:
                    lines.append("")
                    for attachment in message.attachments:
                        lines.append(f"- 图片附件：`{attachment.name}`")
                lines.append("")
                continue
            if message.role != "assistant":
                continue

            heading = "## 助手"
            if message.model:
                heading += f" · `{message.model}`"
            lines.extend([heading, ""])
            # 思考过程用 details 折叠，避免导出后干扰阅读。
            if message.thinking.strip():
                lines.extend(
                    [
                        "<details>",
                        "<summary>思考过程</summary>",
                        "",
                        message.thinking.strip(),
                        "",
                        "</details>",
                        "",
                    ]
                )
            if message.error:
                lines.extend(["**生成失败**", "", message.content.strip() or "_（无内容）_", ""])
            else:
                lines.extend([message.content.strip() or "_（无内容）_", ""])
            # 联网搜索的来源也一起导出，方便回看引用。
            if message.sources:
                lines.extend(["### 联网搜索来源", ""])
                for index, source in enumerate(message.sources, 1):
                    lines.append(f"{index}. [{source.title}]({source.url})")
                lines.append("")
            # 导出时附带 token 数和生成速度。
            if message.eval_count:
                speed = ""
                if message.eval_duration_ns:
                    seconds = message.eval_duration_ns / 1_000_000_000
                    if seconds > 0:
                        speed = f" · {message.eval_count / seconds:.1f} tok/s"
                lines.extend([f"> {message.eval_count} tokens{speed}", ""])

        return "\n".join(lines).strip() + "\n"

    def set_title_from_first_message(self) -> None:
        """用第一条用户消息自动生成侧栏标题。"""
        if self.title and self.title != "新对话":
            return
        for message in self.messages:
            if message.role != "user":
                continue
            if message.content.strip():
                self.title = _clip(message.content, 32)
            elif message.attachments:
                self.title = "图片对话"
            else:
                self.title = "新对话"
            return

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pinned": self.pinned,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Conversation:
        messages_raw = raw.get("messages") or []
        messages = []
        if isinstance(messages_raw, list):
            for item in messages_raw:
                if not isinstance(item, dict):
                    continue
                message = Message.from_dict(item)
                # 中断生成留下的空助手占位消息，重启后直接丢弃。
                if message.role == "assistant" and not message.has_body:
                    continue
                messages.append(message)
        return cls(
            id=str(raw.get("id") or new_id()),
            title=str(raw.get("title") or "新对话"),
            model=str(raw.get("model") or ""),
            messages=messages,
            created_at=_as_float(raw.get("created_at"), time.time()),
            updated_at=_as_float(raw.get("updated_at"), time.time()),
            pinned=bool(raw.get("pinned")),
        )


@dataclass
class Settings:
    """持久化的应用设置。"""

    base_url: str = "http://localhost:11434"
    system_prompt: str = ""
    temperature: float = 0.7
    last_model: str = ""
    deep_thinking: bool = True
    context_size: int = 8192  # num_ctx
    compute_mode: str = "auto"  # auto / cpu / hybrid / gpu
    hybrid_gpu_layers: int = 0  # 0 表示自动一半
    compress_images: bool = True
    web_search: bool = False  # 是否在回答前联网搜索

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "last_model": self.last_model,
            "deep_thinking": self.deep_thinking,
            "context_size": self.context_size,
            "compute_mode": self.compute_mode,
            "hybrid_gpu_layers": self.hybrid_gpu_layers,
            "compress_images": self.compress_images,
            "web_search": self.web_search,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Settings:
        # 对来自磁盘的值做范围收敛，防止损坏配置导致异常。
        temperature = _as_float(raw.get("temperature"), 0.7)
        temperature = max(0.0, min(2.0, temperature))
        compute_mode = str(raw.get("compute_mode") or "auto")
        if compute_mode not in {"auto", "cpu", "hybrid", "gpu"}:
            compute_mode = "auto"
        return cls(
            base_url=str(raw.get("base_url") or "http://localhost:11434").strip(),
            system_prompt=str(raw.get("system_prompt") or ""),
            temperature=temperature,
            last_model=str(raw.get("last_model") or ""),
            deep_thinking=bool(raw.get("deep_thinking", True)),
            context_size=max(2048, min(262144, _as_int(raw.get("context_size"), 8192))),
            compute_mode=compute_mode,
            hybrid_gpu_layers=max(
                0, min(999, _as_int(raw.get("hybrid_gpu_layers"), 0))
            ),
            compress_images=bool(raw.get("compress_images", True)),
            web_search=bool(raw.get("web_search", False)),
        )
