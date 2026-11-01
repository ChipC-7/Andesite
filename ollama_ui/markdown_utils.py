"""把 Markdown 拆成普通文本和围栏代码块。

Flet 的 Markdown 控件不能给单个代码块加按钮，所以回答完成后会按“文本段 +
代码卡片”的方式重新渲染。只拆分已闭合的围栏；流式输出中未闭合的围栏
仍留在普通文本里。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 匹配开围栏、可选语言、主体，以及同字符同长度的闭围栏。
# 主体里更短的围栏不能提前结束代码块。
_FENCE_RE = re.compile(
    r"(?ms)^(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\n]*)\n"
    r"(?P<code>.*?)(?P=fence)[ \t]*$"
)

__all__ = ["MarkdownSegment", "split_fenced_code"]


@dataclass(frozen=True)
class MarkdownSegment:
    # kind 为 text 或 code；code 段的 text 不含围栏。
    kind: str
    text: str
    language: str = ""


def _language(info: str) -> str:
    # 语言信息只取第一个词，并忽略 {.python} 这类属性列表。
    info = (info or "").strip()
    if not info:
        return ""
    language = info.split()[0]
    if language.startswith(("{", ".")) or language.startswith("="):
        return ""
    return language[:24]


def split_fenced_code(text: str) -> list[MarkdownSegment]:
    """按文档顺序返回文本段和代码段；代码段保留原始代码，去掉首尾围栏。"""
    text = text or ""
    if not text:
        return []

    segments: list[MarkdownSegment] = []
    position = 0
    for match in _FENCE_RE.finditer(text):
        # 代码块前面的普通 Markdown。
        before = text[position : match.start()]
        if before:
            segments.append(MarkdownSegment(kind="text", text=before))
        code = match.group("code")
        # 去掉闭围栏前的最后一个换行，方便直接复制。
        if code.endswith("\n"):
            code = code[:-1]
        segments.append(
            MarkdownSegment(
                kind="code",
                text=code,
                language=_language(match.group("info")),
            )
        )
        position = match.end()
    # 最后一个代码块之后的文本。
    remainder = text[position:]
    if remainder:
        segments.append(MarkdownSegment(kind="text", text=remainder))
    return segments
