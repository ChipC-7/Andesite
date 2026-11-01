"""Andesite 主界面：会话侧栏、流式聊天和图片输入。"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import flet as ft

from .client import (
    ModelInfo,
    OllamaClient,
    OllamaError,
    RunningModel,
    build_api_messages,
    gpu_layers_for_mode,
    normalize_base_url,
)
from .images import compress_image
from .markdown_utils import split_fenced_code
from .models import (
    Attachment,
    Conversation,
    Message,
    PendingImage,
    SearchSource,
    Settings,
)
from .storage import Store
from .theme import C, build_theme, markdown_style
from .websearch import WebSearchError, build_search_context, search_web

logger = logging.getLogger(__name__)

APP_VERSION = "0.1.0"
INSTALL_VISION_CMD = "ollama pull llama3.2-vision"
MAX_IMAGES = 8  # 单次最多附加图片数
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 单张图片上限 20 MB
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}
FLUSH_INTERVAL = 0.07  # 流式刷新间隔，约 14 帧/秒
CHAT_IMAGE_DRAG_GROUP = "chat-images"  # 图片拖拽分组

_MIME_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


@dataclass
class MessageView:
    """保存消息各部分的控件引用，流式时只更新需要的控件。"""

    row: ft.Control
    markdown: ft.Markdown | None = None
    status: ft.Text | None = None
    spinner: ft.ProgressRing | None = None
    thinking_box: ft.Container | None = None
    thinking_md: ft.Markdown | None = None
    error_box: ft.Container | None = None
    error_text: ft.Text | None = None
    actions: ft.Row | None = None
    content_host: ft.Container | None = None
    sources_box: ft.Container | None = None
    regenerate_button: ft.TextButton | None = None
    edit_button: ft.TextButton | None = None


class AndesiteApp:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.store = Store()
        self.settings: Settings = self.store.load_settings()
        self.conversations: list[Conversation] = self.store.load_conversations()
        self.current: Conversation | None = self.conversations[0] if self.conversations else None
        self.client = OllamaClient(self.settings.base_url)
        self.models: list[ModelInfo] = []
        self.chat_models: list[ModelInfo] = []
        self.ollama_version: str | None = None
        self.pending_images: list[PendingImage] = []
        self.message_views: dict[str, MessageView] = {}
        self.generating = False
        self.stop_requested = False
        self.refreshing = False
        self.connected = False
        self.active_message_id: str | None = None
        self.runtime_models: list[RunningModel] = []
        # 用户向上翻历史时暂停自动跟随；搜索框和拖拽高亮也保存在这里。
        self._follow_output = True
        self._search_query = ""
        self._drop_highlight = False
        self._md_style = None  # 首次渲染消息时再创建

        # Flet 1.x 通过 services 注册系统服务：文件选择和剪贴板。
        self.file_picker = ft.FilePicker()
        self.page.services.append(self.file_picker)
        self.clipboard = ft.Clipboard()
        self.page.services.append(self.clipboard)

        self._configure_page()
        self._build()
        self._update_header_subtitle()
        self.page.update()
        self.page.run_task(self._startup)

    # ------------------------------------------------------------------
    # 页面与布局
    # ------------------------------------------------------------------
    def _configure_page(self) -> None:
        # 窗口采用无侧边留白的深色布局，并设置最小尺寸。
        page = self.page
        page.title = "Andesite"
        page.padding = 0
        page.spacing = 0
        page.bgcolor = C.BG
        page.theme_mode = ft.ThemeMode.DARK
        # 主题对象构建成本不低，亮/暗主题复用同一份实例。
        theme = build_theme()
        page.theme = theme
        page.dark_theme = theme
        page.window.width = 1180
        page.window.height = 800
        page.window.min_width = 1000
        page.window.min_height = 680
        page.window.bgcolor = C.BG

    def _build(self) -> None:
        # 先构建各区域，再组装成“侧栏 + 主内容”的两栏布局。
        self._build_sidebar()
        self._build_header()
        self._build_banner()
        self._build_chat_area()
        self._build_composer()

        main = ft.Container(
            expand=True,
            gradient=ft.LinearGradient(
                colors=[C.BG_TOP, C.BG],
                begin=ft.Alignment(-1, -1),
                end=ft.Alignment(1, 1),
            ),
            content=ft.Column(
                controls=[
                    self.header,
                    self.banner,
                    # 中间聊天区限制在自己的范围内，避免内容溢到上下栏。
                    ft.Container(
                        expand=True,
                        clip_behavior=ft.ClipBehavior.HARD_EDGE,
                        content=ft.Column(
                            controls=[
                                self.empty_wrap,
                                self.chat_list,
                            ],
                            expand=True,
                            spacing=0,
                        ),
                    ),
                    self.composer_wrap,
                ],
                spacing=0,
                expand=True,
            ),
        )

        # 主内容保持普通两栏布局；拖拽目标只放在输入框周围，
        # 避免 DragTarget 改变整页的约束导致上下栏覆盖聊天区。
        # 首屏只画基础布局；消息列表在 _startup 中稍后构建。
        has_messages = bool(self.current and self.current.messages)
        self.chat_list.visible = has_messages
        self.empty_wrap.visible = not has_messages
        self.header_title.value = (
            self.current.title if has_messages and self.current else "新对话"
        )
        self.delete_button.disabled = self.current is None
        self.page.add(
            ft.Row(
                controls=[self.sidebar, main],
                spacing=0,
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            )
        )
        self._rebuild_sidebar()

    def _build_sidebar(self) -> None:
        # 顶部品牌标识。
        brand_mark = ft.Container(
            width=40,
            height=40,
            border_radius=12,
            gradient=ft.LinearGradient(
                colors=[C.ACCENT, "#38BDF8"],
                begin=ft.Alignment(-1, -1),
                end=ft.Alignment(1, 1),
            ),
            alignment=ft.Alignment(0, 0),
            content=ft.Text("🦙", size=20),
            shadow=ft.BoxShadow(
                blur_radius=18,
                color=ft.Colors.with_opacity(0.22, C.ACCENT),
                offset=ft.Offset(0, 4),
            ),
        )

        brand = ft.Row(
            controls=[
                brand_mark,
                ft.Column(
                    controls=[
                        ft.Text("Andesite", size=17, weight=ft.FontWeight.W_700, color=C.TEXT),
                        ft.Text("本地多模态工作台", size=11, color=C.MUTED),
                    ],
                    spacing=1,
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
            ],
            spacing=11,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # 新建对话按钮。
        self.new_chat_button = ft.Container(
            height=42,
            border_radius=12,
            bgcolor=C.ACCENT,
            on_click=self._new_chat,
            tooltip="开始一个新的对话",
            alignment=ft.Alignment(0, 0),
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.ADD_ROUNDED, size=18, color=C.BG),
                    ft.Text("新建对话", size=13, weight=ft.FontWeight.W_600, color=C.BG),
                ],
                spacing=7,
                alignment=ft.MainAxisAlignment.CENTER,
                tight=True,
            ),
        )

        self.conversation_count = ft.Text("0", size=10, color=C.MUTED_DIM)
        section_label = ft.Row(
            controls=[
                ft.Text("对话历史", size=11, weight=ft.FontWeight.W_600, color=C.MUTED),
                ft.Container(expand=True),
                self.conversation_count,
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # 搜索框按标题和聊天内容实时过滤。
        self.search_field = ft.TextField(
            value="",
            hint_text="搜索对话…",
            hint_style=ft.TextStyle(color=C.MUTED_DIM, size=11),
            prefix_icon=ft.Icons.SEARCH_ROUNDED,
            height=36,
            text_size=11,
            color=C.TEXT,
            bgcolor=C.SURFACE_ALT,
            filled=True,
            border=ft.OutlineInputBorder(
                side=ft.BorderSide(1, C.BORDER_SOFT),
                border_radius=9,
                gap_padding=2,
            ),
            content_padding=ft.Padding.symmetric(horizontal=9, vertical=0),
            on_change=self._on_search_change,
        )

        self.conversation_list = ft.ListView(
            expand=True,
            spacing=3,
            padding=ft.Padding(top=2, bottom=2),
            controls=[],
        )

        # 底部连接状态和设置入口。
        self.connection_dot = ft.Container(
            width=7,
            height=7,
            border_radius=4,
            bgcolor=C.MUTED_DIM,
        )
        self.connection_text = ft.Text(
            "正在连接 Ollama…", size=11, color=C.MUTED, max_lines=2
        )
        self.connection_version = ft.Text(
            "", size=10, color=C.MUTED_DIM, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS
        )
        self.settings_button = ft.TextButton(
            content=ft.Text("设置", size=12, color=C.MUTED),
            icon=ft.Icons.SETTINGS_OUTLINED,
            icon_color=C.MUTED,
            style=ft.ButtonStyle(
                alignment=ft.Alignment(-1, 0),
                padding=ft.Padding.symmetric(horizontal=8, vertical=6),
                overlay_color=ft.Colors.with_opacity(0.05, C.TEXT),
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
            on_click=self._open_settings,
        )

        footer = ft.Container(
            padding=ft.Padding(top=12),
            border=ft.Border(top=ft.BorderSide(1, C.BORDER_SOFT)),
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            self.connection_dot,
                            self.connection_text,
                            ft.Container(expand=True),
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self.connection_version,
                    ft.Container(height=2),
                    self.settings_button,
                ],
                spacing=4,
            ),
        )

        self.sidebar = ft.Container(
            width=282,
            bgcolor=C.SIDEBAR,
            padding=ft.Padding(left=16, top=18, right=16, bottom=12),
            border=ft.Border(right=ft.BorderSide(1, C.BORDER_SOFT)),
            content=ft.Column(
                controls=[
                    brand,
                    ft.Container(height=6),
                    self.new_chat_button,
                    ft.Container(height=10),
                    section_label,
                    self.search_field,
                    self.conversation_list,
                    footer,
                ],
                spacing=8,
                expand=True,
            ),
        )

    def _build_header(self) -> None:
        # 标题下方副标题会动态显示本地/远程状态和计算模式。
        self.header_title = ft.Text(
            "新对话",
            size=18,
            weight=ft.FontWeight.W_600,
            color=C.TEXT,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        self.header_subtitle_icon = ft.Icon(
            ft.Icons.LOCK_OUTLINE, size=11, color=C.MUTED_DIM
        )
        self.header_subtitle_text = ft.Text(
            "本地推理",
            size=11,
            color=C.MUTED_DIM,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
            expand=True,
        )
        self.header_subtitle = ft.Row(
            controls=[
                self.header_subtitle_icon,
                self.header_subtitle_text,
            ],
            spacing=4,
            tight=True,
        )

        # 模型能力标签：视觉模型或纯文本。
        self.model_badge = ft.Container(
            padding=ft.Padding.symmetric(horizontal=9, vertical=5),
            border_radius=20,
            bgcolor=C.SURFACE,
            border=ft.Border.all(1, C.BORDER_SOFT),
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.VISIBILITY_OUTLINED, size=13, color=C.MUTED),
                    ft.Text("选择模型", size=10, color=C.MUTED),
                ],
                spacing=5,
                tight=True,
            ),
        )

        # 仅当模型支持 thinking 时显示的开关。
        self.think_chip = ft.Container(
            visible=False,
            height=28,
            border_radius=14,
            padding=ft.Padding.symmetric(horizontal=9, vertical=4),
            bgcolor=C.SURFACE,
            border=ft.Border.all(1, C.BORDER_SOFT),
            on_click=self._toggle_thinking,
            tooltip="开启后会先输出思考过程，回答可能需要更长时间",
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.PSYCHOLOGY_OUTLINED, size=14, color=C.MUTED),
                    ft.Text("深度思考", size=10, color=C.MUTED),
                    ft.Container(
                        width=28, height=14, border_radius=7, bgcolor=C.BORDER_SOFT
                    ),
                ],
                spacing=6,
                tight=True,
            ),
        )

        self.model_dropdown = ft.Dropdown(
            width=230,
            height=44,
            value=None,
            options=[],
            dense=True,
            filled=True,
            fill_color=C.SURFACE,
            bgcolor=C.SURFACE_ALT,
            border=ft.OutlineInputBorder(
                side=ft.BorderSide(1, C.BORDER_SOFT),
                border_radius=12,
                gap_padding=2,
            ),
            text_size=12,
            color=C.TEXT,
            leading_icon=ft.Icons.MEMORY_ROUNDED,
            hint_text="加载模型中…",
            hint_style=ft.TextStyle(color=C.MUTED_DIM, size=12),
            content_padding=ft.Padding.symmetric(horizontal=10, vertical=0),
            disabled=True,
            on_select=self._on_model_select,
        )
        # 刷新模型、删除当前对话。
        self.refresh_button = ft.IconButton(
            icon=ft.Icons.REFRESH_ROUNDED,
            icon_size=18,
            icon_color=C.MUTED,
            tooltip="刷新模型列表",
            on_click=self._refresh_clicked,
            style=ft.ButtonStyle(
                overlay_color=ft.Colors.with_opacity(0.06, C.TEXT),
                shape=ft.RoundedRectangleBorder(radius=10),
            ),
        )
        self.delete_button = ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE,
            icon_size=18,
            icon_color=C.MUTED,
            tooltip="删除当前对话",
            disabled=True,
            on_click=self._delete_current,
            style=ft.ButtonStyle(
                overlay_color=ft.Colors.with_opacity(0.06, C.DANGER),
                shape=ft.RoundedRectangleBorder(radius=10),
            ),
        )

        self.header = ft.Container(
            padding=ft.Padding(left=28, top=15, right=20, bottom=13),
            bgcolor=C.BG_TOP,
            border=ft.Border(bottom=ft.BorderSide(1, C.BORDER_SOFT)),
            content=ft.Row(
                controls=[
                    ft.Column(
                        controls=[self.header_title, self.header_subtitle],
                        spacing=3,
                        expand=True,
                    ),
                    self.model_badge,
                    self.think_chip,
                    self.model_dropdown,
                    self.refresh_button,
                    self.delete_button,
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    def _build_banner(self) -> None:
        # 顶部横幅，用于离线、无模型等提示。
        self.banner = ft.Container(
            visible=False,
            margin=ft.Margin(left=28, top=14, right=28, bottom=0),
            padding=ft.Padding.symmetric(horizontal=14, vertical=12),
            border_radius=12,
            border=ft.Border.all(1, C.BORDER),
            bgcolor=C.SURFACE,
        )

    def _build_chat_area(self) -> None:
        # 消息列表：交给 Flet 原生的 auto_scroll 处理贴底和上翻暂停，
        # 不要手动 update 列表，否则滚动位置会被重置。
        self.chat_list = ft.ListView(
            expand=True,
            spacing=22,
            padding=ft.Padding(left=28, top=24, right=28, bottom=20),
            auto_scroll=True,
            # Flet 默认用 1 秒动画；流式输出需要瞬时贴底。
            auto_scroll_animation=0,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            controls=[],
        )
        self.empty_wrap = ft.Container(
            expand=True,
            alignment=ft.Alignment(0, 0),
            padding=ft.Padding(left=28, right=28, top=10, bottom=10),
            content=self._build_empty_state(),
        )

    def _set_follow_output(self, follow: bool) -> None:
        # auto_scroll 保持开启，Flet 会自动在用户上翻时暂停、回底时恢复；
        # 这里只同步状态，不调用 chat_list.update()，避免重置滚动位置。
        self._follow_output = follow
        if follow:
            self.chat_list.auto_scroll = True

    def _build_empty_state(self) -> ft.Control:
        mark = ft.Container(
            width=76,
            height=76,
            border_radius=22,
            gradient=ft.LinearGradient(
                colors=[C.ACCENT, "#38BDF8"],
                begin=ft.Alignment(-1, -1),
                end=ft.Alignment(1, 1),
            ),
            alignment=ft.Alignment(0, 0),
            content=ft.Text("🦙", size=38),
            shadow=ft.BoxShadow(
                blur_radius=28,
                color=ft.Colors.with_opacity(0.18, C.ACCENT),
                offset=ft.Offset(0, 8),
            ),
        )

        self.empty_vision_note = ft.Text(
            "",
            size=11,
            color=C.WARNING,
            text_align=ft.TextAlign.CENTER,
            visible=False,
        )

        suggestions = ft.Container(
            height=126,
            content=ft.Row(
                controls=[
                    self._suggestion_card(
                        icon=ft.Icons.IMAGE_SEARCH,
                        title="分析图片",
                        description="选择图片或直接粘贴截图",
                        on_click=self._pick_images,
                    ),
                    self._suggestion_card(
                        icon=ft.Icons.CODE_ROUNDED,
                        title="写一段代码",
                        description="生成、解释或调试代码",
                        on_click=lambda e: self._fill_prompt("请用 Python 帮我实现："),
                    ),
                    self._suggestion_card(
                        icon=ft.Icons.NOTES_ROUNDED,
                        title="总结文字",
                        description="粘贴长文，提炼关键信息",
                        on_click=lambda e: self._fill_prompt("请总结以下内容：\n"),
                    ),
                ],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
        )

        self.empty_note = ft.Text(
            "在终端里没法发图片？这里可以选择或粘贴截图，"
            "再交给 Ollama 的视觉模型。",
            size=13,
            color=C.MUTED,
            text_align=ft.TextAlign.CENTER,
            width=500,
        )

        content = ft.Column(
            controls=[
                mark,
                ft.Container(height=16),
                ft.Text(
                    "让本地模型看懂图片",
                    size=26,
                    weight=ft.FontWeight.W_700,
                    color=C.TEXT,
                    text_align=ft.TextAlign.CENTER,
                ),
                ft.Container(height=6),
                self.empty_note,
                ft.Container(height=22),
                suggestions,
                ft.Container(height=10),
                self.empty_vision_note,
            ],
            spacing=0,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            width=640,
        )
        return content

    def _suggestion_card(
        self,
        *,
        icon: ft.IconData,
        title: str,
        description: str,
        on_click: Any,
    ) -> ft.Control:
        return ft.Container(
            expand=1,
            height=126,
            bgcolor=C.SURFACE,
            border=ft.Border.all(1, C.BORDER_SOFT),
            border_radius=14,
            padding=ft.Padding.symmetric(horizontal=14, vertical=13),
            on_click=on_click,
            content=ft.Column(
                controls=[
                    ft.Container(
                        width=32,
                        height=32,
                        border_radius=9,
                        bgcolor=C.ACCENT_SOFT,
                        alignment=ft.Alignment(0, 0),
                        content=ft.Icon(icon, size=17, color=C.ACCENT),
                    ),
                    ft.Container(height=2),
                    ft.Text(
                        title,
                        size=12,
                        weight=ft.FontWeight.W_600,
                        color=C.TEXT,
                        max_lines=1,
                    ),
                    ft.Text(
                        description,
                        size=10,
                        color=C.MUTED,
                        max_lines=2,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                ],
                spacing=4,
                alignment=ft.MainAxisAlignment.START,
            ),
        )

    def _build_composer(self) -> None:
        # 待发送图片缩略图和模型能力警告。
        self.attachments_bar = ft.Row(
            controls=[],
            spacing=8,
            run_spacing=8,
            wrap=True,
            visible=False,
        )
        self.attachment_warning = ft.Text(
            "",
            size=10,
            color=C.WARNING,
            visible=False,
        )

        # 输入框支持 Enter 发送、Shift+Enter 换行和粘贴截图。
        self.input = ft.TextField(
            multiline=True,
            min_lines=1,
            max_lines=6,
            shift_enter=True,
            value="",
            hint_text="输入消息，或粘贴截图；也可把聊天图片拖到左侧图片按钮…",
            hint_style=ft.TextStyle(color=C.MUTED_DIM, size=13),
            border=ft.NoInputBorder(),
            filled=False,
            bgcolor=ft.Colors.TRANSPARENT,
            text_size=14,
            color=C.TEXT,
            text_style=ft.TextStyle(height=1.45),
            cursor_color=C.ACCENT,
            selection_color=ft.Colors.with_opacity(0.28, C.ACCENT),
            content_padding=ft.Padding.symmetric(horizontal=3, vertical=6),
            autofocus=True,
            on_submit=self._on_submit,
            on_paste_files=self._on_paste_files,
        )

        self.attach_button = ft.IconButton(
            icon=ft.Icons.ADD_PHOTO_ALTERNATE,
            icon_size=19,
            icon_color=C.MUTED,
            tooltip="添加图片（也可以按 Ctrl+V 粘贴截图，或把聊天图片拖到这个按钮）",
            width=36,
            height=36,
            style=ft.ButtonStyle(
                overlay_color=ft.Colors.with_opacity(0.06, C.TEXT),
                shape=ft.RoundedRectangleBorder(radius=9),
            ),
            on_click=self._pick_images,
        )
        # 拖拽接收区只包住图片按钮，避免 DragTarget 把整个输入框压窄。
        self.attach_drop_target = ft.DragTarget(
            group=CHAT_IMAGE_DRAG_GROUP,
            on_will_accept=self._on_composer_drag_will_accept,
            on_leave=self._on_composer_drag_leave,
            on_accept=self._on_composer_image_drop,
            content=self.attach_button,
        )

        self.send_button = ft.FilledButton(
            width=40,
            height=40,
            tooltip="发送（Enter）",
            content=ft.Icon(ft.Icons.ARROW_UPWARD_ROUNDED, size=19, color=C.BG),
            style=ft.ButtonStyle(
                bgcolor=C.ACCENT,
                color=C.BG,
                padding=0,
                shape=ft.RoundedRectangleBorder(radius=20),
                overlay_color=ft.Colors.with_opacity(0.12, C.BG),
            ),
            on_click=self._send_clicked,
        )

        # 联网搜索快捷开关：开启后每次提问前先用 Bing 搜网页。
        self.web_search_button = ft.Container(
            width=30,
            height=30,
            border_radius=9,
            alignment=ft.Alignment(0, 0),
            on_click=self._toggle_web_search,
            content=ft.Icon(ft.Icons.PUBLIC, size=17, color=C.MUTED),
        )
        self._update_web_search_button()

        # 上下文占用提示，请求完成后更新。
        self.composer_hint = ft.Text(
            f"上下文 — / {self.settings.context_size:,}",
            size=10,
            color=C.MUTED_DIM,
        )
        self.keyboard_hint = ft.Text(
            "Enter 发送 · Shift+Enter 换行",
            size=10,
            color=C.MUTED_DIM,
        )

        self.composer = ft.Container(
            bgcolor=C.SURFACE,
            border=ft.Border.all(1, C.BORDER),
            border_radius=16,
            padding=ft.Padding(left=13, top=10, right=10, bottom=8),
            content=ft.Column(
                controls=[
                    self.attachments_bar,
                    self.attachment_warning,
                    self.input,
                    ft.Row(
                        controls=[
                            self.attach_drop_target,
                            self.web_search_button,
                            self.composer_hint,
                            ft.Container(expand=True),
                            self.keyboard_hint,
                            self.send_button,
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ],
                spacing=4,
                tight=True,
            ),
        )

        # 输入框直接放在容器里，保持横向铺满聊天区域。
        self.composer_wrap = ft.Container(
            padding=ft.Padding(left=28, top=8, right=28, bottom=18),
            bgcolor=C.BG,
            content=self.composer,
        )

    # ------------------------------------------------------------------
    # 启动与模型发现
    # ------------------------------------------------------------------
    async def _startup(self) -> None:
        # 窗口居中放到后台，不阻塞首次连接。
        self.page.run_task(self._center_window)
        # 先让基础界面完成首帧，再构建消息和 Markdown 控件。
        await asyncio.sleep(0.05)
        self._render_conversation()
        await self.refresh_models(notify=False)

    async def _center_window(self) -> None:
        try:
            await self.page.window.center()
        except Exception:  # window management is best-effort (e.g. browser mode)
            logger.debug("Could not center window", exc_info=True)

    async def _refresh_clicked(self, e: Any = None) -> None:
        await self.refresh_models(notify=True)

    async def refresh_models(self, notify: bool = True) -> None:
        # 防止刷新按钮被连续点击时重复请求。
        if self.refreshing:
            return
        self.refreshing = True
        self._set_connection("loading", "正在连接 Ollama…")
        self.refresh_button.disabled = True
        self.page.update(self.refresh_button, self.connection_dot, self.connection_text)
        try:
            # 模型列表和版本号互不依赖，并行请求缩短启动等待。
            models, version = await asyncio.gather(
                self.client.list_models(),
                self.client.get_version(),
            )
        except OllamaError as exc:
            # 连接失败时清空模型/运行时信息，并显示离线引导。
            self.connected = False
            self.models = []
            self.chat_models = []
            self.ollama_version = None
            self.runtime_models = []
            self._set_connection("offline", "无法连接 Ollama")
            self._show_offline_banner(str(exc))
            self._refresh_model_options()
            if notify:
                self._snack(str(exc), tone="error")
            logger.warning("Ollama connection failed: %s", exc)
            return
        finally:
            self.refreshing = False
            self.refresh_button.disabled = self.generating
            self.page.update(self.refresh_button)

        self.connected = True
        self.models = models
        self.chat_models = [model for model in models if model.is_chat_model]
        self.ollama_version = version
        self._select_default_model()
        self._set_connection(
            "online" if self.chat_models else "warning",
            f"已连接 · {len(self.chat_models)} 个对话模型"
            if self.chat_models
            else "已连接 · 没有对话模型",
        )
        self.connection_version.value = (
            f"Ollama {self.ollama_version}" if self.ollama_version else ""
        )
        self._refresh_model_options()
        self._update_model_meta()
        self._update_connection_banner()
        self.empty_vision_note.visible = not any(
            model.supports_vision for model in self.chat_models
        )
        if self.empty_vision_note.visible:
            self.empty_vision_note.value = (
                "还没有视觉模型。终端执行  ollama pull llama3.2-vision  即可解锁图片对话。"
            )
        self.page.update(
            self.connection_version,
            self.empty_vision_note,
        )
        # 模型列表就绪后，再异步查询 /api/ps 的实际显存占用。
        self.page.run_task(self._refresh_runtime_info)
        if notify:
            self._snack(
                f"模型列表已刷新，共 {len(self.models)} 个模型。", tone="success"
            )

    def _select_default_model(self) -> None:
        # 选择优先级：当前会话模型 -> 上次使用 -> 视觉模型 -> 第一个模型。
        names = {model.name for model in self.chat_models}
        preferred: list[str] = []
        if self.current and self.current.model:
            preferred.append(self.current.model)
        if self.settings.last_model:
            preferred.append(self.settings.last_model)
        vision = next((m.name for m in self.chat_models if m.supports_vision), "")
        if vision:
            preferred.append(vision)
        if self.chat_models:
            preferred.append(self.chat_models[0].name)

        chosen = next((name for name in preferred if name in names), "")
        if self.current:
            if self.current.model not in names:
                self.current.model = chosen
                self.current.touch()
                self._persist_conversations()
            chosen = self.current.model or chosen
        if chosen and chosen != self.settings.last_model:
            self.settings.last_model = chosen
            self._save_settings()

    def _refresh_model_options(self) -> None:
        self.model_dropdown.options = [
            ft.DropdownOption(key=model.name, text=model.name)
            for model in self.chat_models
        ]
        selected = ""
        if self.current and self.current.model in {m.name for m in self.chat_models}:
            selected = self.current.model
        elif self.settings.last_model in {m.name for m in self.chat_models}:
            selected = self.settings.last_model
        elif self.chat_models:
            selected = self.chat_models[0].name
        self.model_dropdown.value = selected or None
        self.model_dropdown.disabled = self.generating or not bool(self.chat_models)
        self.model_dropdown.hint_text = (
            "选择一个模型" if self.chat_models else "没有可对话的模型"
        )
        self._update_model_meta()
        self.page.update(self.model_dropdown)

    def _update_connection_banner(self) -> None:
        # 没有对话模型时显示引导，否则隐藏横幅。
        if not self.connected:
            return  # the offline banner was already shown
        if not self.chat_models:
            command_box = ft.Container(
                bgcolor=C.SURFACE_ALT,
                border_radius=8,
                padding=ft.Padding.symmetric(horizontal=10, vertical=7),
                content=ft.Text(
                    INSTALL_VISION_CMD,
                    size=11,
                    color=C.ACCENT_TEXT,
                    font_family="monospace",
                    selectable=True,
                ),
            )
            self._show_banner(
                icon=ft.Icons.MODEL_TRAINING,
                title="还没有可对话的模型",
                body=ft.Column(
                    controls=[
                        ft.Text(
                            "当前只有嵌入模型或模型列表为空。先在终端下载一个模型，"
                            "视觉对话推荐：",
                            size=11,
                            color=C.MUTED,
                        ),
                        command_box,
                    ],
                    spacing=7,
                    tight=True,
                ),
                tone="warning",
                actions=[
                    ft.TextButton(
                        content=ft.Text("复制命令", size=11, color=C.WARNING),
                        on_click=self._copy_install_command,
                    ),
                    ft.TextButton(
                        content=ft.Text("刷新模型", size=11, color=C.TEXT_SOFT),
                        on_click=self._refresh_clicked,
                    ),
                ],
            )
        else:
            self.banner.visible = False
            self.banner.update()

    def _show_offline_banner(self, detail: str) -> None:
        self._show_banner(
            icon=ft.Icons.CLOUD_OFF_OUTLINED,
            title="连接不到 Ollama",
            body=ft.Text(
                f"{detail}\n如果服务未启动，请在终端运行：ollama serve",
                size=11,
                color=C.MUTED,
                selectable=True,
            ),
            tone="error",
            actions=[
                ft.TextButton(
                    content=ft.Text("重试连接", size=11, color=C.DANGER),
                    on_click=self._refresh_clicked,
                ),
                ft.TextButton(
                    content=ft.Text("设置地址", size=11, color=C.TEXT_SOFT),
                    on_click=self._open_settings,
                ),
            ],
        )

    def _show_banner(
        self,
        *,
        icon: ft.IconData,
        title: str,
        body: ft.Control,
        tone: str,
        actions: list[ft.Control] | None = None,
    ) -> None:
        palette = {
            "error": (C.DANGER_SOFT, C.DANGER_BORDER, C.DANGER),
            "warning": (C.WARNING_SOFT, C.WARNING_BORDER, C.WARNING),
            "info": (C.SURFACE_ALT, C.BORDER, C.ACCENT),
        }
        bgcolor, border_color, icon_color = palette.get(
            tone, (C.SURFACE_ALT, C.BORDER, C.ACCENT)
        )
        self.banner.bgcolor = bgcolor
        self.banner.border = ft.Border.all(1, border_color)
        self.banner.content = ft.Row(
            controls=[
                ft.Icon(icon, size=19, color=icon_color),
                ft.Column(
                    controls=[
                        ft.Text(
                            title,
                            size=12,
                            weight=ft.FontWeight.W_600,
                            color=C.TEXT,
                        ),
                        body,
                    ],
                    spacing=3,
                    expand=True,
                ),
                *(actions or []),
            ],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
        self.banner.visible = True
        self.banner.update()

    def _set_connection(self, state: str, text: str) -> None:
        # 侧栏底部连接状态：在线/警告/离线/加载中。
        colors = {
            "online": C.SUCCESS,
            "warning": C.WARNING,
            "offline": C.DANGER,
            "loading": C.MUTED,
        }
        self.connection_dot.bgcolor = colors.get(state, C.MUTED_DIM)
        self.connection_text.value = text
        try:
            self.connection_dot.update()
            self.connection_text.update()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 模型辅助
    # ------------------------------------------------------------------
    def _model_info(self, name: str | None) -> ModelInfo | None:
        if not name:
            return None
        return next((model for model in self.models if model.name == name), None)

    def _current_model_supports_vision(self) -> bool:
        info = self._model_info(self.model_dropdown.value)
        return bool(info and info.supports_vision)

    def _gpu_layers(self, info: ModelInfo | None) -> int | None:
        """把计算模式设置转换成 Ollama 的 num_gpu 参数。"""
        return gpu_layers_for_mode(
            self.settings.compute_mode,
            info.layer_count if info else 0,
            self.settings.hybrid_gpu_layers,
        )

    def _update_model_meta(self) -> None:
        # 同步模型能力标签、附件警告、thinking 开关和计算模式副标题。
        name = self.model_dropdown.value or ""
        info = self._model_info(name)
        self.model_badge.visible = bool(name)
        if not name:
            self.model_badge.content = ft.Row(
                controls=[
                    ft.Icon(ft.Icons.VISIBILITY_OUTLINED, size=13, color=C.MUTED),
                    ft.Text("选择模型", size=10, color=C.MUTED),
                ],
                spacing=5,
                tight=True,
            )
            self.model_badge.bgcolor = C.SURFACE
            self.model_badge.border = ft.Border.all(1, C.BORDER_SOFT)
            self.model_badge.tooltip = None
        elif info and info.supports_vision:
            self.model_badge.content = ft.Row(
                controls=[
                    ft.Icon(ft.Icons.VISIBILITY_ROUNDED, size=13, color=C.ACCENT),
                    ft.Text("视觉模型", size=10, color=C.ACCENT),
                ],
                spacing=5,
                tight=True,
            )
            self.model_badge.bgcolor = C.ACCENT_SOFT
            self.model_badge.border = ft.Border.all(1, C.ACCENT_BORDER)
            self.model_badge.tooltip = "该模型支持图片输入"
        else:
            self.model_badge.content = ft.Row(
                controls=[
                    ft.Icon(ft.Icons.TEXT_FIELDS_ROUNDED, size=13, color=C.MUTED),
                    ft.Text("纯文本", size=10, color=C.MUTED),
                ],
                spacing=5,
                tight=True,
            )
            self.model_badge.bgcolor = C.SURFACE
            self.model_badge.border = ft.Border.all(1, C.BORDER_SOFT)
            self.model_badge.tooltip = "该模型不支持图片输入"
        try:
            self.model_badge.update()
        except Exception:
            pass
        self._refresh_attachment_warning()
        self._update_think_control()
        self._update_header_subtitle()

    def _update_think_control(self) -> None:
        # 模型不支持 thinking 时隐藏开关。
        info = self._model_info(self.model_dropdown.value)
        supported = bool(info and info.supports_thinking)
        self.think_chip.visible = supported
        if not supported:
            try:
                self.think_chip.update()
            except Exception:
                pass
            return
        on = self.settings.deep_thinking
        color = C.ACCENT if on else C.MUTED
        self.think_chip.bgcolor = C.ACCENT_SOFT if on else C.SURFACE
        self.think_chip.border = ft.Border.all(
            1, C.ACCENT_BORDER if on else C.BORDER_SOFT
        )
        self.think_chip.opacity = 0.55 if self.generating else 1.0
        self.think_chip.tooltip = (
            "深度思考已开启，点击关闭。模型会先输出思考过程，回答可能需要更长时间。"
            if on
            else "深度思考已关闭，点击开启。模型会先进行推理再回答。"
        )
        self.think_chip.content = ft.Row(
            controls=[
                ft.Icon(
                    ft.Icons.PSYCHOLOGY_ROUNDED
                    if on
                    else ft.Icons.PSYCHOLOGY_OUTLINED,
                    size=14,
                    color=color,
                ),
                ft.Text("深度思考", size=10, color=color),
                ft.Stack(
                    width=28,
                    height=14,
                    controls=[
                        ft.Container(
                            width=28,
                            height=14,
                            border_radius=7,
                            bgcolor=C.ACCENT if on else C.BORDER_SOFT,
                        ),
                        ft.Container(
                            width=10,
                            height=10,
                            border_radius=5,
                            bgcolor=C.BG if on else C.MUTED,
                            left=16 if on else 2,
                            top=2,
                        ),
                    ],
                ),
            ],
            spacing=6,
            tight=True,
        )
        try:
            self.think_chip.update()
        except Exception:
            logger.debug("Could not update thinking control", exc_info=True)

    def _toggle_thinking(self, e: Any = None) -> None:
        if self.generating:
            return
        info = self._model_info(self.model_dropdown.value)
        if not info or not info.supports_thinking:
            return
        self.settings.deep_thinking = not self.settings.deep_thinking
        self._save_settings()
        self._update_think_control()

    def _update_web_search_button(self) -> None:
        # 开启时用强调色高亮，关闭时保持低干扰。
        enabled = self.settings.web_search
        self.web_search_button.bgcolor = (
            C.ACCENT_SOFT if enabled else ft.Colors.TRANSPARENT
        )
        self.web_search_button.border = ft.Border.all(
            1, C.ACCENT_BORDER if enabled else C.BORDER_SOFT
        )
        self.web_search_button.tooltip = (
            "联网搜索已开启（使用 Bing，点击关闭）"
            if enabled
            else "联网搜索已关闭（点击开启）"
        )
        self.web_search_button.content = ft.Icon(
            ft.Icons.PUBLIC,
            size=17,
            color=C.ACCENT if enabled else C.MUTED,
        )
        try:
            self.web_search_button.update()
        except Exception:
            pass

    def _toggle_web_search(self, e: Any = None) -> None:
        self.settings.web_search = not self.settings.web_search
        self._save_settings()
        self._update_web_search_button()
        # 顶部的数据去向提示需要随开关即时更新。
        self._update_header_subtitle()
        if self.settings.web_search:
            self._snack(
                "联网搜索已开启：问题会发送给 Bing 做网页搜索。", tone="success"
            )
        else:
            self._snack("联网搜索已关闭。", tone="info")

    def _on_model_select(self, e: Any = None) -> None:
        if self.generating:
            return
        name = getattr(e.control, "value", None) or self.model_dropdown.value
        if not name:
            return
        self.settings.last_model = name
        self._save_settings()
        if self.current:
            self.current.model = name
            self.current.touch()
            self._persist_conversations()
        self._update_model_meta()
        self.page.run_task(self._refresh_runtime_info)

    def _refresh_attachment_warning(self) -> None:
        if not self.pending_images:
            self.attachment_warning.visible = False
        else:
            if self._current_model_supports_vision():
                self.attachment_warning.visible = False
            else:
                any_vision = any(m.supports_vision for m in self.chat_models)
                if any_vision:
                    self.attachment_warning.value = (
                        "当前模型不支持图片输入，请切换到视觉模型后再发送。"
                    )
                else:
                    self.attachment_warning.value = (
                        "当前模型不支持图片输入，且没有检测到视觉模型。"
                        f"可执行  {INSTALL_VISION_CMD}  安装。"
                    )
                self.attachment_warning.visible = True
        try:
            self.attachment_warning.update()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 侧栏会话列表
    # ------------------------------------------------------------------
    def _on_search_change(self, e: Any) -> None:
        # 搜索词变化时只重建列表，不重建搜索框，避免焦点丢失。
        self._search_query = (e.control.value or "").strip()
        self._rebuild_sidebar()

    def _rebuild_sidebar(self) -> None:
        # 先搜索过滤，再按“置顶优先、最近活动其次”排序。
        query = self._search_query
        filtered = [
            conversation
            for conversation in self.conversations
            if not query or conversation.matches(query)
        ]
        ordered = sorted(
            filtered,
            key=lambda conversation: (
                0 if conversation.pinned else 1,
                -conversation.last_message_at,
            ),
        )
        tiles: list[ft.Control] = []
        if not self.conversations:
            tiles.append(
                ft.Container(
                    padding=ft.Padding(left=10, top=12, right=10, bottom=12),
                    content=ft.Text(
                        "还没有对话。\n从下面输入框开始吧。",
                        size=11,
                        color=C.MUTED_DIM,
                        text_align=ft.TextAlign.LEFT,
                    ),
                )
            )
        elif not ordered:
            tiles.append(
                ft.Container(
                    padding=ft.Padding(left=10, top=12, right=10, bottom=12),
                    content=ft.Text(
                        "没有找到匹配的对话。",
                        size=11,
                        color=C.MUTED_DIM,
                    ),
                )
            )
        for conversation in ordered:
            tiles.append(self._conversation_tile(conversation))
        self.conversation_list.controls = tiles
        self.conversation_count.value = (
            f"{len(filtered)}/{len(self.conversations)}"
            if query
            else str(len(self.conversations))
        )
        try:
            self.conversation_list.update()
            self.conversation_count.update()
            self.new_chat_button.opacity = 0.55 if self.generating else 1.0
            self.new_chat_button.update()
        except Exception:
            pass

    def _conversation_tile(self, conversation: Conversation) -> ft.Control:
        # 当前会话用左侧强调条和浅色底突出显示。
        active = self.current is not None and conversation.id == self.current.id
        accent_bar = ft.Container(
            width=3,
            height=30,
            border_radius=2,
            bgcolor=C.ACCENT if active else ft.Colors.TRANSPARENT,
        )
        title_controls: list[ft.Control] = []
        if conversation.pinned:
            title_controls.append(
                ft.Icon(ft.Icons.PUSH_PIN, size=11, color=C.ACCENT)
            )
        title_controls.append(
            ft.Text(
                conversation.title or "新对话",
                size=12,
                weight=ft.FontWeight.W_600 if active else ft.FontWeight.W_500,
                color=C.TEXT if active else C.TEXT_SOFT,
                max_lines=1,
                overflow=ft.TextOverflow.ELLIPSIS,
                expand=True,
            )
        )
        # 每个会话的更多操作：重命名、置顶、导出、删除。
        menu = ft.PopupMenuButton(
            items=[
                ft.PopupMenuItem(
                    content=ft.Text("重命名", size=12, color=C.TEXT_SOFT),
                    icon=ft.Icons.EDIT_OUTLINED,
                    on_click=lambda e, c=conversation: self._rename_conversation(c),
                ),
                ft.PopupMenuItem(
                    content=ft.Text(
                        "取消置顶" if conversation.pinned else "置顶",
                        size=12,
                        color=C.TEXT_SOFT,
                    ),
                    icon=ft.Icons.PUSH_PIN,
                    on_click=lambda e, c=conversation: self._toggle_pin(c),
                ),
                ft.PopupMenuItem(
                    content=ft.Text("导出 Markdown", size=12, color=C.TEXT_SOFT),
                    icon=ft.Icons.IOS_SHARE_ROUNDED,
                    on_click=lambda e, c=conversation: self.page.run_task(
                        self._export_conversation, c
                    ),
                ),
                ft.PopupMenuItem(
                    content=ft.Text("删除", size=12, color=C.DANGER),
                    icon=ft.Icons.DELETE_OUTLINE,
                    on_click=lambda e, c=conversation: self._confirm_delete(c),
                ),
            ],
            icon=ft.Icons.MORE_HORIZ,
            icon_color=C.MUTED_DIM,
            icon_size=16,
            tooltip="更多操作",
            width=30,
            height=30,
            padding=0,
            menu_position=ft.PopupMenuPosition.UNDER,
            style=ft.ButtonStyle(
                overlay_color=ft.Colors.with_opacity(0.05, C.TEXT),
                shape=ft.RoundedRectangleBorder(radius=7),
            ),
        )
        return ft.Container(
            padding=ft.Padding(left=8, top=9, right=4, bottom=9),
            border_radius=10,
            bgcolor=C.ACCENT_SOFT if active else ft.Colors.TRANSPARENT,
            on_click=lambda e, cid=conversation.id: self._switch_conversation(cid),
            content=ft.Row(
                controls=[
                    accent_bar,
                    ft.Column(
                        controls=[
                            ft.Row(
                                controls=title_controls,
                                spacing=5,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                            ft.Text(
                                conversation.preview,
                                size=10,
                                color=C.MUTED if active else C.MUTED_DIM,
                                max_lines=1,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ],
                        spacing=2,
                        expand=True,
                    ),
                    menu,
                ],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    # ------------------------------------------------------------------
    # 会话生命周期
    # ------------------------------------------------------------------
    def _new_chat(self, e: Any = None) -> None:
        # 新建对话只清空当前视图，真正保存发生在第一条消息发送时。
        if self.generating:
            self._snack("请先停止当前生成。", tone="warning")
            return
        self.current = None
        self.pending_images.clear()
        self._refresh_attachments()
        self.input.value = ""
        self._set_follow_output(True)
        self._render_conversation()
        self._rebuild_sidebar()
        self._select_model_from_settings()
        self.page.run_task(self.input.focus)

    def _select_model_from_settings(self) -> None:
        names = {model.name for model in self.chat_models}
        name = self.settings.last_model if self.settings.last_model in names else ""
        if not name and self.chat_models:
            name = next(
                (m.name for m in self.chat_models if m.supports_vision),
                self.chat_models[0].name,
            )
        self.model_dropdown.value = name or None
        self._update_model_meta()
        try:
            self.model_dropdown.update()
        except Exception:
            pass

    def _switch_conversation(self, conversation_id: str) -> None:
        # 生成中禁止切换，避免流式内容写进另一个会话。
        if self.generating:
            self._snack("请先停止当前生成，再切换对话。", tone="warning")
            return
        conversation = next(
            (item for item in self.conversations if item.id == conversation_id), None
        )
        if conversation is None or (
            self.current is not None and conversation.id == self.current.id
        ):
            return
        self.current = conversation
        self.pending_images.clear()
        self._refresh_attachments()
        self.input.value = ""
        self._set_follow_output(True)
        names = {model.name for model in self.chat_models}
        if conversation.model in names:
            self.model_dropdown.value = conversation.model
        else:
            self._select_model_from_settings()
        self._render_conversation()
        self._rebuild_sidebar()
        self._update_model_meta()
        try:
            self.model_dropdown.update()
        except Exception:
            pass

    def _rename_conversation(self, conversation: Conversation) -> None:
        # 手动重命名后，后续不会再被自动标题覆盖。
        if self.generating:
            self._snack("请先停止当前生成。", tone="warning")
            return
        name_field = ft.TextField(
            value=conversation.title,
            label="对话名称",
            autofocus=True,
            width=420,
            text_size=13,
            color=C.TEXT,
            label_style=ft.TextStyle(color=C.MUTED, size=11),
            border=ft.OutlineInputBorder(
                side=ft.BorderSide(1, C.BORDER),
                border_radius=12,
            ),
            filled=True,
            fill_color=C.SURFACE_ALT,
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=10),
        )

        def save(event: Any) -> None:
            title = (name_field.value or "").strip()
            if not title:
                name_field.error_text = "名称不能为空"
                name_field.update()
                return
            conversation.title = title[:80]
            self._persist_conversations()
            self.page.pop_dialog()
            self._rebuild_sidebar()
            if self.current is not None and self.current.id == conversation.id:
                self._update_header()
            self._snack("对话已重命名。", tone="success")

        dialog = ft.AlertDialog(
            modal=True,
            bgcolor=C.SURFACE,
            shape=ft.RoundedRectangleBorder(radius=18),
            title=ft.Text(
                "重命名对话", size=17, weight=ft.FontWeight.W_600, color=C.TEXT
            ),
            content=name_field,
            actions=[
                ft.TextButton(
                    content=ft.Text("取消", size=12, color=C.MUTED),
                    on_click=lambda event: self.page.pop_dialog(),
                ),
                ft.FilledButton(
                    content=ft.Text("保存", size=12),
                    style=ft.ButtonStyle(
                        bgcolor=C.ACCENT,
                        color=C.BG,
                        padding=ft.Padding.symmetric(horizontal=16, vertical=10),
                        shape=ft.RoundedRectangleBorder(radius=10),
                    ),
                    on_click=save,
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dialog)

    def _toggle_pin(self, conversation: Conversation) -> None:
        # 置顶状态会持久化，并立即重排侧栏。
        conversation.pinned = not conversation.pinned
        self._persist_conversations()
        self._rebuild_sidebar()
        self._snack(
            "对话已置顶。" if conversation.pinned else "已取消置顶。",
            tone="success",
        )

    @staticmethod
    def _safe_filename(title: str) -> str:
        cleaned = re.sub(r'[\\/:*?"<>|\r\n]+', "_", title or "").strip(" ._")
        return (cleaned[:48] or "conversation")

    async def _export_conversation(self, conversation: Conversation) -> None:
        # 调用系统保存对话框；src_bytes 会由 Flet 写入目标文件。
        data = conversation.to_markdown().encode("utf-8")
        try:
            path = await self.file_picker.save_file(
                dialog_title="导出 Markdown",
                file_name=f"{self._safe_filename(conversation.title)}.md",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["md"],
                src_bytes=data,
            )
        except Exception as exc:  # noqa: BLE001 - native dialogs are platform-specific
            logger.exception("Could not export conversation")
            self._snack(f"导出失败：{exc}", tone="error")
            return
        if not path:
            return
        # 桌面端通常会写入 src_bytes；这里兜底只返回路径的平台。
        target = Path(path)
        try:
            if not target.exists():
                target.write_bytes(data)
        except OSError:
            logger.warning("Could not write exported conversation", exc_info=True)
        self._snack(f"已导出：{target.name}", tone="success")

    def _delete_current(self, e: Any = None) -> None:
        if self.generating:
            self._snack("请先停止当前生成。", tone="warning")
            return
        if self.current is not None:
            self._confirm_delete(self.current)

    def _confirm_delete(self, conversation: Conversation) -> None:
        if self.generating:
            self._snack("请先停止当前生成。", tone="warning")
            return

        def confirm(event: Any) -> None:
            self.page.pop_dialog()
            self._delete_conversation(conversation)

        dialog = ft.AlertDialog(
            modal=True,
            bgcolor=C.SURFACE,
            shape=ft.RoundedRectangleBorder(radius=18),
            title=ft.Text(
                "删除这个对话？", size=17, weight=ft.FontWeight.W_600, color=C.TEXT
            ),
            content=ft.Text(
                f"「{conversation.title}」的记录和其中的图片会从本机删除，"
                "此操作无法撤销。",
                size=12,
                color=C.MUTED,
            ),
            actions=[
                ft.TextButton(
                    content=ft.Text("取消", size=12, color=C.MUTED),
                    on_click=lambda event: self.page.pop_dialog(),
                ),
                ft.TextButton(
                    content=ft.Text("删除", size=12, color=C.DANGER),
                    on_click=confirm,
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dialog)

    def _delete_conversation(self, conversation: Conversation) -> None:
        # 删除会话时同步清理其图片；只有当前会话被删才切换界面。
        was_current = self.current is not None and self.current.id == conversation.id
        self.store.delete_conversation_images(conversation)
        self.conversations = [
            item for item in self.conversations if item.id != conversation.id
        ]
        if was_current:
            ordered = sorted(
                self.conversations,
                key=lambda item: (0 if item.pinned else 1, -item.last_message_at),
            )
            self.current = ordered[0] if ordered else None
        self._persist_conversations()
        self._rebuild_sidebar()
        if not was_current:
            self._snack("对话已删除。", tone="success")
            return

        self.pending_images.clear()
        self._refresh_attachments()
        self.input.value = ""
        self._render_conversation()
        if self.current:
            names = {model.name for model in self.chat_models}
            if self.current.model in names:
                self.model_dropdown.value = self.current.model
            else:
                self._select_model_from_settings()
            self._update_model_meta()
        else:
            self._select_model_from_settings()
        self._snack("对话已删除。", tone="success")

    def _render_conversation(self, update: bool = True) -> None:
        # 重建整段聊天视图；切换会话或编辑消息后调用。
        self.message_views.clear()
        self.chat_list.controls = []
        has_messages = bool(self.current and self.current.messages)
        self._set_follow_output(True)
        if self.current:
            for message in self.current.messages:
                view = self._build_message_view(message)
                self.message_views[message.id] = view
                self.chat_list.controls.append(view.row)
        self.chat_list.visible = has_messages
        self.empty_wrap.visible = not has_messages
        self._update_header()
        self._update_context_hint()
        if update:
            try:
                self.page.update()
            except Exception:
                logger.debug("Could not update page after rendering", exc_info=True)

    def _compute_summary(self) -> str:
        # 顶部显示的“计算模式 + 实际 GPU 层数”。
        mode = self.settings.compute_mode
        info = self._model_info(getattr(self.model_dropdown, "value", None))
        layers = info.layer_count if info else 0
        custom = self.settings.hybrid_gpu_layers
        if mode == "cpu":
            return "计算：纯 CPU"
        if mode == "gpu":
            return "计算：纯 GPU"
        if mode == "hybrid":
            num_gpu = gpu_layers_for_mode(mode, layers, custom)
            if layers > 0:
                summary = f"计算：混合 {num_gpu}/{layers} 层"
            else:
                summary = f"计算：混合 {num_gpu} 层"
            if custom > 0:
                summary += " · 自定义"
            return summary
        return "计算：自动"

    def _model_runtime(self) -> RunningModel | None:
        selected = getattr(self.model_dropdown, "value", None)
        if not selected:
            return None
        for running in self.runtime_models:
            if running.name == selected:
                return running
        # Ollama 可能省略 latest 标签，按基础名再匹配一次。
        base = selected.split(":", 1)[0]
        return next(
            (
                running
                for running in self.runtime_models
                if running.name.split(":", 1)[0] == base
            ),
            None,
        )

    def _runtime_summary(self) -> str:
        # 根据 /api/ps 的显存占用给出实际加载情况。
        running = self._model_runtime()
        if running is None or running.size <= 0:
            return ""
        share = running.vram_share
        if share >= 0.97:
            return f"实际全部在显存 {running.size_vram / 1e9:.1f} GB"
        if share <= 0.03:
            return "实际在 CPU"
        return (
            f"实际显存 {running.size_vram / 1e9:.1f}/"
            f"{running.size / 1e9:.1f} GB"
        )

    def _compute_tooltip(self) -> str:
        mode = self.settings.compute_mode
        info = self._model_info(getattr(self.model_dropdown, "value", None))
        layers = info.layer_count if info else 0
        num_gpu = gpu_layers_for_mode(
            mode, layers, self.settings.hybrid_gpu_layers
        )
        text = (
            "num_gpu 由 Ollama 自动决定（请求不传该参数）"
            if num_gpu is None
            else f"请求使用 num_gpu={num_gpu}"
        )
        if layers:
            text += f"；模型共 {layers} 层"
        running = self._model_runtime()
        if running is not None and running.size > 0:
            text += (
                f"；Ollama 报告显存占用 {running.size_vram / 1e9:.1f} GB / "
                f"总占用 {running.size / 1e9:.1f} GB"
            )
        return text

    def _update_header_subtitle(self) -> None:
        # 副标题显示隐私提示、计算模式和实际显存；远程 Ollama 与联网搜索都要提示数据去向。
        host = (urlparse(self.settings.base_url).hostname or "").lower()
        is_local = host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
        searching = self.settings.web_search
        privacy_parts: list[str] = []
        tooltip_parts: list[str] = []
        if is_local:
            # 本地模式下只有开启联网搜索时才需要在顶部提示。
            self.header_subtitle_icon.visible = searching
            if searching:
                self.header_subtitle_icon.name = ft.Icons.PUBLIC
                self.header_subtitle_icon.color = C.WARNING
                self.header_subtitle_text.color = C.WARNING
                privacy_parts.append("联网搜索 · 搜索词会发送到 Bing")
                tooltip_parts.append("联网搜索已开启：当前问题会发送给 Bing 做网页搜索。")
            else:
                self.header_subtitle_text.color = C.MUTED_DIM
            if hasattr(self, "empty_note"):
                self.empty_note.value = (
                    "在终端里没法发图片？这里可以选择或粘贴截图，"
                    "再交给 Ollama 的视觉模型。"
                    + (
                        "聊天内容只发送到本机 Ollama；联网搜索的词会发送给 Bing。"
                        if searching
                        else "所有内容只留在本机。"
                    )
                )
        else:
            # 连接远程地址时提示消息会发送到该服务器；联网搜索再额外提示 Bing。
            self.header_subtitle_icon.visible = True
            self.header_subtitle_icon.name = ft.Icons.CLOUD_OUTLINED
            self.header_subtitle_icon.color = C.WARNING
            self.header_subtitle_text.color = C.WARNING
            privacy_parts.append("远程 Ollama · 数据会发送到该服务器")
            tooltip_parts.append("消息和图片会发送到远程 Ollama 服务器。")
            if searching:
                privacy_parts.append("搜索词也会发送到 Bing")
                tooltip_parts.append("联网搜索会把当前问题发送给 Bing。")
            if hasattr(self, "empty_note"):
                self.empty_note.value = (
                    "当前连接的是远程 Ollama 地址，图片和文本会发送到该服务器"
                    + (
                        "；联网搜索还会把搜索词发送给 Bing，而不是只留在本机。"
                        if searching
                        else "，而不是只留在本机。"
                    )
                )
        privacy = " · ".join(privacy_parts)
        parts = [privacy, self._compute_summary(), self._runtime_summary()]
        self.header_subtitle_text.value = " · ".join(
            part for part in parts if part
        )
        tooltip_parts.append(self._compute_tooltip())
        tooltip = "\n".join(tooltip_parts)
        self.header_subtitle_text.tooltip = tooltip
        self.header_subtitle.tooltip = tooltip
        try:
            self.header_subtitle_icon.update()
            self.header_subtitle_text.update()
            if hasattr(self, "empty_note"):
                self.empty_note.update()
        except Exception:
            logger.debug("Could not update privacy subtitle", exc_info=True)

    async def _refresh_runtime_info(self) -> None:
        """查询 /api/ps，显示模型实际加载到显存的情况。"""
        try:
            self.runtime_models = await self.client.running_models()
        except Exception:  # noqa: BLE001 - runtime details are best-effort
            logger.debug("Could not refresh running models", exc_info=True)
            return
        self._update_header_subtitle()

    def _update_header(self) -> None:
        self.header_title.value = (
            self.current.title if self.current and self.current.messages else "新对话"
        )
        self.delete_button.disabled = self.current is None or self.generating
        try:
            self.header_title.update()
            self.delete_button.update()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 消息渲染
    # ------------------------------------------------------------------
    def _ensure_md_style(self) -> ft.MarkdownStyleSheet:
        # Markdown 样式模块导入较慢，只在真正渲染消息时加载。
        if self._md_style is None:
            self._md_style = markdown_style()
        return self._md_style

    def _build_message_view(self, message: Message) -> MessageView:
        self._ensure_md_style()
        if message.role == "user":
            return self._build_user_message_view(message)
        return self._build_assistant_message_view(message)

    def _humanize_error_content(self, content: str, model: str) -> str:
        """把旧版本保存的原始 JSON 错误转换成可读中文。"""
        if not content or "exceed_context_size_error" not in content:
            return content
        json_start = content.find("{")
        if json_start < 0:
            return content
        try:
            return OllamaClient._error_from_body(
                content[json_start:].encode("utf-8"), 400, model
            )
        except Exception:
            return content

    def _schedule_encoded_image(self, image: ft.RawImage, data: bytes) -> None:
        """通过专用数据通道显示大图。

        普通 ``Image.src`` 走常规 Flet 协议，几 MB 的截图可能静默失败；
        ``RawImage`` 用独立数据通道，能稳定渲染完整截图。
        """
        self.page.run_task(self._render_encoded_image, image, data)

    async def _render_encoded_image(self, image: ft.RawImage, data: bytes) -> None:
        try:
            await image.render_encoded(data)
        except Exception:
            logger.warning("Could not render image in the UI", exc_info=True)

    def _build_user_message_view(self, message: Message) -> MessageView:
        # 用户消息靠右显示，附带图片和复制/编辑操作。
        contents: list[ft.Control] = []
        if message.attachments:
            contents.append(self._attachments_row(message.attachments))
        markdown: ft.Markdown | None = None
        if message.content.strip():
            markdown = ft.Markdown(
                value=message.content,
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
                code_theme=ft.MarkdownCodeTheme.ATOM_ONE_DARK,
                md_style_sheet=self._md_style,
                auto_follow_links=True,
                soft_line_break=True,
                shrink_wrap=True,
            )
            contents.append(markdown)
        bubble = ft.Container(
            padding=ft.Padding.symmetric(horizontal=15, vertical=12),
            border_radius=ft.BorderRadius(top_left=18, top_right=18, bottom_left=18, bottom_right=7),
            bgcolor=C.USER_BG,
            border=ft.Border.all(1, C.USER_BORDER),
            content=ft.Column(controls=contents, spacing=9, tight=True),
        )
        copy_button = ft.TextButton(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.CONTENT_COPY_ROUNDED, size=12, color=C.MUTED),
                    ft.Text("复制", size=10, color=C.MUTED),
                ],
                spacing=4,
                tight=True,
            ),
            tooltip="复制这条消息",
            visible=bool(message.content.strip()),
            style=ft.ButtonStyle(
                padding=ft.Padding.symmetric(horizontal=6, vertical=1),
                overlay_color=ft.Colors.with_opacity(0.05, C.TEXT),
                shape=ft.RoundedRectangleBorder(radius=7),
            ),
            on_click=lambda e, msg=message: self.page.run_task(
                self._copy_message, msg
            ),
        )
        edit_button = ft.TextButton(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.EDIT_OUTLINED, size=12, color=C.MUTED),
                    ft.Text("编辑并重发", size=10, color=C.MUTED),
                ],
                spacing=4,
                tight=True,
            ),
            tooltip="修改这条消息，并从该处重新生成回答",
            visible=not self.generating,
            style=ft.ButtonStyle(
                padding=ft.Padding.symmetric(horizontal=6, vertical=1),
                overlay_color=ft.Colors.with_opacity(0.05, C.TEXT),
                shape=ft.RoundedRectangleBorder(radius=7),
            ),
            on_click=lambda e, msg=message: self._edit_message(msg),
        )
        actions = ft.Row(
            controls=[copy_button, edit_button],
            spacing=2,
            tight=True,
            alignment=ft.MainAxisAlignment.END,
        )
        column = ft.Column(
            controls=[bubble, actions],
            spacing=3,
            tight=True,
            horizontal_alignment=ft.CrossAxisAlignment.END,
        )
        # 左侧留白 + 右侧弹性区域，把用户气泡限制在约 80% 宽度内；
        # 否则长文本会把整行撑到窗口外，导致消息“消失”。
        row = ft.Row(
            controls=[
                ft.Container(expand=1),
                ft.Container(expand=4, content=column),
            ],
            spacing=0,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
        return MessageView(
            row=row,
            markdown=markdown,
            actions=actions,
            edit_button=edit_button,
        )

    def _build_assistant_message_view(self, message: Message) -> MessageView:
        # 助手消息包含头像、思考折叠、正文、错误框和底部操作。
        avatar = ft.Container(
            width=34,
            height=34,
            border_radius=11,
            bgcolor=C.ACCENT_SOFT,
            border=ft.Border.all(1, C.ACCENT_BORDER),
            alignment=ft.Alignment(0, 0),
            content=ft.Text("🦙", size=17),
        )

        model_name = message.model or "助手"
        status = ft.Text(
            self._message_status_text(message),
            size=10,
            color=C.MUTED_DIM,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )
        spinner = ft.ProgressRing(
            width=12,
            height=12,
            stroke_width=2,
            color=C.ACCENT,
            visible=False,
        )
        header = ft.Row(
            controls=[
                spinner,
                ft.Text(
                    model_name,
                    size=11,
                    weight=ft.FontWeight.W_600,
                    color=C.MUTED,
                    max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                status,
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        thinking_md = ft.Markdown(
            value=message.thinking,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
            code_theme=ft.MarkdownCodeTheme.ATOM_ONE_DARK,
            md_style_sheet=self._md_style,
            shrink_wrap=False,
        )
        thinking_box = ft.Container(
            visible=bool(message.thinking.strip()),
            border_radius=10,
            content=ft.ExpansionTile(
                title=ft.Text("思考过程", size=11, color=C.MUTED),
                controls=[thinking_md],
                expanded=False,
                dense=True,
                maintain_state=True,
                show_trailing_icon=True,
                min_tile_height=36,
                tile_padding=ft.Padding.symmetric(horizontal=10, vertical=0),
                controls_padding=ft.Padding(left=10, right=10, bottom=10, top=0),
                bgcolor=C.SURFACE_ALT,
                collapsed_bgcolor=C.SURFACE_ALT,
                text_color=C.MUTED,
                icon_color=C.MUTED,
                shape=ft.RoundedRectangleBorder(radius=10),
                collapsed_shape=ft.RoundedRectangleBorder(radius=10),
                visual_density=ft.VisualDensity.COMPACT,
            ),
        )

        markdown = ft.Markdown(
            value=message.content,
            visible=not message.error,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
            code_theme=ft.MarkdownCodeTheme.ATOM_ONE_DARK,
            md_style_sheet=self._md_style,
            auto_follow_links=True,
            shrink_wrap=False,
        )
        content_host = ft.Container(
            visible=not message.error,
            content=(
                markdown
                if message.error
                else (self._rich_content(message.content) or markdown)
            ),
        )

        error_text = ft.Text(
            self._humanize_error_content(message.content, message.model or "")
            if message.error
            else message.content,
            size=12,
            color=C.DANGER,
            selectable=True,
        )
        error_box = ft.Container(
            visible=message.error,
            bgcolor=C.DANGER_SOFT,
            border=ft.Border.all(1, C.DANGER_BORDER),
            border_radius=10,
            padding=ft.Padding.symmetric(horizontal=12, vertical=9),
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.ERROR_OUTLINE, size=16, color=C.DANGER),
                    ft.Container(content=error_text, expand=True),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
        )
        sources_box = self._build_sources_box(message.sources)

        is_last_assistant = bool(
            self.current
            and self.current.messages
            and self.current.messages[-1].id == message.id
            and self.current.messages[-1].role == "assistant"
        )
        copy_button = ft.TextButton(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.CONTENT_COPY_ROUNDED, size=13, color=C.MUTED),
                    ft.Text("复制", size=10, color=C.MUTED),
                ],
                spacing=4,
                tight=True,
            ),
            tooltip="复制这条回答",
            visible=not message.error,
            style=ft.ButtonStyle(
                padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                overlay_color=ft.Colors.with_opacity(0.05, C.TEXT),
                shape=ft.RoundedRectangleBorder(radius=7),
            ),
            on_click=lambda e, msg=message: self.page.run_task(
                self._copy_message, msg
            ),
        )
        regenerate_button = ft.TextButton(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.REPLAY_ROUNDED, size=13, color=C.MUTED),
                    ft.Text(
                        "重试" if message.error else "重新生成",
                        size=10,
                        color=C.MUTED,
                    ),
                ],
                spacing=4,
                tight=True,
            ),
            tooltip="用同样的上下文再生成一次",
            visible=is_last_assistant,
            style=ft.ButtonStyle(
                padding=ft.Padding.symmetric(horizontal=6, vertical=2),
                overlay_color=ft.Colors.with_opacity(0.05, C.TEXT),
                shape=ft.RoundedRectangleBorder(radius=7),
            ),
            on_click=lambda e, msg=message: self._regenerate_message(msg),
        )
        actions = ft.Row(
            controls=[copy_button, regenerate_button],
            spacing=2,
            visible=bool(message.content.strip() or message.thinking.strip() or message.error),
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        column = ft.Column(
            controls=[
                header,
                thinking_box,
                content_host,
                error_box,
                sources_box,
                actions,
            ],
            spacing=8,
            tight=True,
        )
        row = ft.Row(
            controls=[
                avatar,
                ft.Container(content=column, expand=True, padding=ft.Padding(top=1)),
            ],
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
        return MessageView(
            row=row,
            markdown=markdown,
            content_host=content_host,
            sources_box=sources_box,
            status=status,
            spinner=spinner,
            thinking_box=thinking_box,
            thinking_md=thinking_md,
            error_box=error_box,
            error_text=error_text,
            actions=actions,
            regenerate_button=regenerate_button,
        )

    def _sources_content(self, sources: list[SearchSource]) -> ft.Control:
        # 用 Markdown 列出可点击来源；替换会破坏语法的字符。
        lines = []
        for index, source in enumerate(sources, 1):
            title = source.title.replace("[", "［").replace("]", "］")
            url = source.url.replace("(", "%28").replace(")", "%29")
            lines.append(f"{index}. [{title}]({url})")
        return ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.PUBLIC, size=13, color=C.ACCENT),
                        ft.Text(
                            f"联网搜索来源 · {len(sources)}",
                            size=10,
                            color=C.MUTED,
                            weight=ft.FontWeight.W_600,
                        ),
                    ],
                    spacing=6,
                    tight=True,
                ),
                ft.Markdown(
                    value="\n".join(lines),
                    selectable=True,
                    extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
                    md_style_sheet=self._ensure_md_style(),
                    auto_follow_links=True,
                    shrink_wrap=False,
                ),
            ],
            spacing=5,
            tight=True,
        )

    def _build_sources_box(self, sources: list[SearchSource]) -> ft.Container:
        # 没有来源时整块隐藏，不占用回答下方空间。
        return ft.Container(
            visible=bool(sources),
            bgcolor=C.SURFACE_ALT,
            border=ft.Border.all(1, C.BORDER_SOFT),
            border_radius=10,
            padding=ft.Padding.symmetric(horizontal=12, vertical=9),
            content=self._sources_content(sources),
        )

    def _sync_sources_view(
        self, view: MessageView | None, message: Message
    ) -> None:
        # 搜索完成后原地刷新来源框，避免重建整条消息。
        if not view or not view.sources_box:
            return
        view.sources_box.content = self._sources_content(message.sources)
        view.sources_box.visible = bool(message.sources)
        try:
            view.sources_box.update()
        except Exception:
            logger.debug("Could not update sources box", exc_info=True)

    def _rich_content(self, content: str) -> ft.Control | None:
        """把完成的回答渲染成文本段 + 可单独复制的代码卡片。"""
        segments = split_fenced_code(content)
        if not segments or not any(seg.kind == "code" for seg in segments):
            return None
        controls: list[ft.Control] = []
        for segment in segments:
            if segment.kind == "code":
                controls.append(
                    self._code_block_card(segment.language, segment.text)
                )
            elif segment.text.strip():
                controls.append(
                    ft.Markdown(
                        value=segment.text,
                        selectable=True,
                        extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
                        code_theme=ft.MarkdownCodeTheme.ATOM_ONE_DARK,
                        md_style_sheet=self._md_style,
                        auto_follow_links=True,
                        shrink_wrap=False,
                    )
                )
        if not controls:
            return None
        return ft.Column(controls=controls, spacing=9, tight=True)

    def _code_block_card(self, language: str, code: str) -> ft.Control:
        # 代码卡片顶部显示语言/行数，右侧提供复制按钮。
        label = language.upper() if language else "CODE"
        lines = len(code.splitlines()) or 1
        copy_button = ft.TextButton(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.CONTENT_COPY_ROUNDED, size=12, color=C.MUTED),
                    ft.Text("复制", size=10, color=C.MUTED),
                ],
                spacing=4,
                tight=True,
            ),
            tooltip="复制这个代码块",
            style=ft.ButtonStyle(
                padding=ft.Padding.symmetric(horizontal=6, vertical=1),
                overlay_color=ft.Colors.with_opacity(0.05, C.TEXT),
                shape=ft.RoundedRectangleBorder(radius=6),
            ),
            on_click=lambda e, value=code: self.page.run_task(
                self._copy_text, value, "代码已复制"
            ),
        )
        header = ft.Row(
            controls=[
                ft.Icon(ft.Icons.CODE_ROUNDED, size=13, color=C.MUTED),
                ft.Text(
                    f"{label} · {lines} 行",
                    size=10,
                    color=C.MUTED_DIM,
                    max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                ft.Container(expand=True),
                copy_button,
            ],
            spacing=6,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        code_text = ft.Text(
            code or " ",
            size=13,
            color=C.TEXT_SOFT,
            font_family="monospace",
            selectable=True,
            no_wrap=True,
        )
        body = ft.Row(
            controls=[code_text],
            scroll=ft.ScrollMode.AUTO,
            spacing=0,
        )
        return ft.Container(
            bgcolor=C.CODE_BG,
            border=ft.Border.all(1, C.BORDER_SOFT),
            border_radius=10,
            padding=ft.Padding(left=12, top=6, right=8, bottom=12),
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            content=ft.Column(
                controls=[header, body],
                spacing=4,
                tight=True,
            ),
        )

    def _attachments_row(self, attachments: list[Attachment]) -> ft.Control:
        # 图片缩略图；右上角拖拽手柄可把图片复用为待发送附件。
        tiles = []
        for attachment in attachments:
            data = self._read_attachment(attachment)
            image: ft.Control
            if data:
                raw_image = ft.RawImage(
                    width=150,
                    height=150,
                    fit=ft.BoxFit.COVER,
                    filter_quality=ft.FilterQuality.MEDIUM,
                    ready_timeout=20,
                )
                self._schedule_encoded_image(raw_image, data)
                image = raw_image
            else:
                image = ft.Container(
                    width=150,
                    height=150,
                    bgcolor=C.SURFACE_ALT,
                    border_radius=10,
                    alignment=ft.Alignment(0, 0),
                    content=ft.Column(
                        controls=[
                            ft.Icon(ft.Icons.BROKEN_IMAGE_OUTLINED, size=22, color=C.MUTED),
                            ft.Text("图片已丢失", size=10, color=C.MUTED),
                        ],
                        spacing=4,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                )
            tile = ft.Container(
                border_radius=10,
                clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                on_click=(
                    (lambda e, d=data, n=attachment.name: self._preview_image(d, n))
                    if data
                    else None
                ),
                tooltip=f"{attachment.name}（可拖到输入框复用）",
                content=image,
            )
            feedback = ft.Container(
                width=96,
                height=96,
                border_radius=14,
                bgcolor=ft.Colors.with_opacity(0.94, C.SURFACE_ALT),
                border=ft.Border.all(2, C.ACCENT),
                alignment=ft.Alignment(0, 0),
                shadow=ft.BoxShadow(
                    blur_radius=16,
                    color=ft.Colors.with_opacity(0.35, C.BG),
                    offset=ft.Offset(0, 6),
                ),
                content=ft.Column(
                    controls=[
                        ft.Icon(
                            ft.Icons.DRAG_HANDLE_ROUNDED,
                            size=26,
                            color=C.ACCENT,
                        ),
                        ft.Text(
                            "拖到输入框",
                            size=10,
                            color=C.ACCENT_TEXT,
                        ),
                    ],
                    spacing=2,
                    tight=True,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            )
            if data:
                # 独立手柄避免拖拽与列表滚动、图片点击预览冲突。
                handle_surface = ft.Container(
                    width=30,
                    height=30,
                    border_radius=15,
                    bgcolor=ft.Colors.with_opacity(0.86, C.SURFACE),
                    border=ft.Border.all(
                        1, ft.Colors.with_opacity(0.62, C.ACCENT)
                    ),
                    alignment=ft.Alignment(0, 0),
                    tooltip="拖动到输入框复用这张图片",
                    shadow=ft.BoxShadow(
                        blur_radius=8,
                        color=ft.Colors.with_opacity(0.32, C.BG),
                        offset=ft.Offset(0, 2),
                    ),
                    scale=1.0,
                    animate_scale=120,
                    on_hover=self._on_drag_handle_hover,
                    content=ft.Icon(
                        ft.Icons.DRAG_HANDLE_ROUNDED,
                        size=16,
                        color=C.ACCENT_TEXT,
                    ),
                )
                drag_handle = ft.Draggable(
                    group=CHAT_IMAGE_DRAG_GROUP,
                    data=attachment.id,
                    content=handle_surface,
                    content_feedback=feedback,
                )
                tiles.append(
                    ft.Stack(
                        width=150,
                        height=150,
                        controls=[
                            tile,
                            ft.Container(
                                right=7,
                                top=7,
                                width=30,
                                height=30,
                                content=drag_handle,
                            ),
                        ],
                    )
                )
            else:
                tiles.append(tile)
        return ft.Row(controls=tiles, spacing=8, run_spacing=8, wrap=True, tight=True)

    def _message_status_text(self, message: Message) -> str:
        if message.error:
            return ""
        bits: list[str] = []
        if message.eval_count and message.eval_duration_ns:
            seconds = message.eval_duration_ns / 1_000_000_000
            if seconds > 0:
                bits.append(f"{message.eval_count} tokens")
                bits.append(f"{message.eval_count / seconds:.1f} tok/s")
        elif message.eval_count:
            bits.append(f"{message.eval_count} tokens")
        return " · ".join(bits)

    def _update_context_hint(self) -> None:
        # 用最近一次请求的 prompt tokens 估算上下文占用。
        context_size = self.settings.context_size
        prompt_tokens = 0
        if self.current:
            for message in reversed(self.current.messages):
                if message.role == "assistant" and message.prompt_eval_count:
                    prompt_tokens = message.prompt_eval_count
                    break
        if prompt_tokens:
            percent = prompt_tokens / context_size if context_size else 0.0
            self.composer_hint.value = (
                f"上下文 {prompt_tokens:,} / {context_size:,} · {percent * 100:.0f}%"
            )
            if percent >= 0.95:
                self.composer_hint.color = C.DANGER
            elif percent >= 0.8:
                self.composer_hint.color = C.WARNING
            else:
                self.composer_hint.color = C.MUTED_DIM
        else:
            self.composer_hint.value = f"上下文 — / {context_size:,}"
            self.composer_hint.color = C.MUTED_DIM
        try:
            self.composer_hint.update()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 输入框与图片
    # ------------------------------------------------------------------
    def _fill_prompt(self, prompt: str) -> None:
        # 点击空状态示例时把提示词填入输入框并聚焦。
        self.input.value = prompt
        try:
            self.input.update()
        except Exception:
            pass
        self.page.run_task(self.input.focus)

    def _on_drag_handle_hover(self, e: Any) -> None:
        # 悬停时放大并提亮拖拽手柄。
        hovered = bool(getattr(e, "data", False))
        control = getattr(e, "control", None)
        if control is None:
            return
        control.bgcolor = ft.Colors.with_opacity(
            0.96 if hovered else 0.86,
            C.ACCENT_SOFT if hovered else C.SURFACE,
        )
        control.border = ft.Border.all(
            1,
            C.ACCENT
            if hovered
            else ft.Colors.with_opacity(0.62, C.ACCENT),
        )
        control.scale = 1.08 if hovered else 1.0
        control.shadow = ft.BoxShadow(
            blur_radius=12 if hovered else 8,
            color=ft.Colors.with_opacity(
                0.3 if hovered else 0.32,
                C.ACCENT if hovered else C.BG,
            ),
            offset=ft.Offset(0, 2),
        )
        try:
            control.update()
        except Exception:
            logger.debug("Could not update drag handle hover", exc_info=True)

    def _set_composer_drop_highlight(self, active: bool) -> None:
        # 拖拽悬停时给输入框加强调色边框。
        if active == self._drop_highlight:
            return
        self._drop_highlight = active
        self.composer.border = ft.Border.all(
            1, C.ACCENT if active else C.BORDER
        )
        self.composer.bgcolor = C.SURFACE
        try:
            self.composer.update()
        except Exception:
            logger.debug("Could not update drop highlight", exc_info=True)

    def _on_composer_drag_will_accept(self, e: Any) -> None:
        self._set_composer_drop_highlight(bool(getattr(e, "accept", False)))

    def _on_composer_drag_leave(self, e: Any) -> None:
        self._set_composer_drop_highlight(False)

    def _attachment_by_id(self, attachment_id: str) -> Attachment | None:
        # 拖拽只传附件 ID，需要从所有会话里反查对象。
        for conversation in self.conversations:
            for message in conversation.messages:
                for attachment in message.attachments:
                    if attachment.id == attachment_id:
                        return attachment
        return None

    def _on_composer_image_drop(self, e: Any) -> None:
        # 把聊天里已有的图片读出来，重新加入待发送附件。
        self._set_composer_drop_highlight(False)
        if self.generating:
            self._snack("请先停止当前生成，再把图片拖进来。", tone="warning")
            return
        source = getattr(e, "src", None)
        attachment_id = str(
            getattr(source, "data", None) or getattr(e, "data", None) or ""
        )
        attachment = self._attachment_by_id(attachment_id)
        if attachment is None:
            self._snack("无法识别拖入的图片。", tone="warning")
            return
        data = self._read_attachment(attachment)
        if not data:
            self._snack("这张图片的文件已丢失。", tone="warning")
            return
        self._add_pending_images(
            [
                PendingImage(
                    name=attachment.name,
                    mime=attachment.mime,
                    data=data,
                )
            ]
        )

    async def _pick_images(self, e: Any = None) -> None:
        # 调用系统文件选择器，读取图片字节后进入统一的添加流程。
        try:
            files = await self.file_picker.pick_files(
                dialog_title="选择图片",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=sorted(ALLOWED_IMAGE_EXTENSIONS),
                allow_multiple=True,
                with_data=True,
            )
        except Exception as exc:  # noqa: BLE001 - native dialog failures are platform issues
            logger.exception("File picker failed")
            self._snack(f"无法打开文件选择器：{exc}", tone="error")
            return
        if not files:
            return
        new_images: list[PendingImage] = []
        for file in files:
            data = getattr(file, "bytes", None)
            if not data and getattr(file, "path", None):
                try:
                    data = Path(file.path).read_bytes()
                except OSError:
                    data = None
            if not data:
                self._snack(f"无法读取 {file.name}。", tone="error")
                continue
            mime = self._guess_mime(file.name, "")
            if not mime.startswith("image/"):
                self._snack(f"{file.name} 不是图片文件。", tone="warning")
                continue
            new_images.append(PendingImage(name=file.name, mime=mime, data=data))
        self._add_pending_images(new_images)

    def _on_paste_files(self, e: Any) -> None:
        # TextField 粘贴文件时触发，目前只接受图片。
        images: list[PendingImage] = []
        for file in getattr(e, "files", []) or []:
            data = getattr(file, "bytes", None)
            if not data:
                continue
            name = getattr(file, "name", "") or "pasted-image.png"
            mime = getattr(file, "mime_type", "") or self._guess_mime(name, "")
            if not mime.startswith("image/"):
                self._snack("暂时只支持粘贴图片。", tone="warning")
                continue
            images.append(PendingImage(name=name, mime=mime, data=data))
        self._add_pending_images(images)

    @staticmethod
    def _guess_mime(name: str, fallback: str) -> str:
        extension = Path(name).suffix.lower()
        return _MIME_BY_EXTENSION.get(extension) or fallback or "application/octet-stream"

    def _add_pending_images(self, images: list[PendingImage]) -> None:
        # 统一检查数量/大小，并按设置压缩后加入待发送列表。
        added = 0
        compressed = 0
        for image in images:
            if len(self.pending_images) >= MAX_IMAGES:
                self._snack(f"最多一次附加 {MAX_IMAGES} 张图片。", tone="warning")
                break
            if len(image.data) > MAX_IMAGE_BYTES:
                self._snack(
                    f"{image.name} 超过 20 MB，未添加。", tone="warning"
                )
                continue
            original_size = len(image.data)
            if self.settings.compress_images:
                data, mime, name = compress_image(
                    image.data, image.name, image.mime
                )
                if len(data) < original_size:
                    image = PendingImage(name=name, mime=mime, data=data)
                    compressed += 1
            self.pending_images.append(image)
            added += 1
        if added:
            if not self._current_model_supports_vision():
                self._snack(
                    "图片已添加。当前模型不支持图片，发送前请切换到视觉模型。",
                    tone="warning",
                )
            elif compressed:
                self._snack(
                    f"已添加 {added} 张图片，并压缩了 {compressed} 张以节省上下文。",
                    tone="success",
                )
            else:
                self._snack(f"已添加 {added} 张图片。", tone="success")
        self._refresh_attachments()

    def _refresh_attachments(self) -> None:
        # 重建待发送图片缩略图；这里同样使用 RawImage 数据通道。
        controls: list[ft.Control] = []
        for image in self.pending_images:
            raw_image = ft.RawImage(
                width=76,
                height=76,
                fit=ft.BoxFit.COVER,
                ready_timeout=20,
            )
            self._schedule_encoded_image(raw_image, image.data)
            stack = ft.Stack(
                width=76,
                height=76,
                controls=[
                    ft.Container(
                        width=76,
                        height=76,
                        border_radius=10,
                        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                        content=raw_image,
                    ),
                    ft.Container(
                        width=20,
                        height=20,
                        right=3,
                        top=3,
                        border_radius=10,
                        bgcolor=ft.Colors.with_opacity(0.72, "#000000"),
                        alignment=ft.Alignment(0, 0),
                        tooltip="移除图片",
                        on_click=lambda e, image_id=image.id: self._remove_pending_image(image_id),
                        content=ft.Icon(ft.Icons.CLOSE_ROUNDED, size=12, color=C.TEXT),
                    ),
                ],
            )
            controls.append(stack)
        self.attachments_bar.controls = controls
        self.attachments_bar.visible = bool(controls)
        try:
            self.attachments_bar.update()
        except Exception:
            pass
        self._refresh_attachment_warning()

    def _remove_pending_image(self, image_id: str) -> None:
        self.pending_images = [
            image for image in self.pending_images if image.id != image_id
        ]
        self._refresh_attachments()

    def _preview_image(self, data: bytes | None, name: str) -> None:
        # 点击缩略图时用 RawImage 数据通道打开大图预览。
        if not data:
            self._snack("图片文件已丢失。", tone="error")
            return
        raw_image = ft.RawImage(
            expand=True,
            fit=ft.BoxFit.CONTAIN,
            filter_quality=ft.FilterQuality.MEDIUM,
            ready_timeout=20,
        )
        dialog = ft.AlertDialog(
            modal=False,
            bgcolor=C.SURFACE,
            shape=ft.RoundedRectangleBorder(radius=18),
            title=ft.Text(
                name,
                size=14,
                weight=ft.FontWeight.W_600,
                color=C.TEXT,
                max_lines=1,
                overflow=ft.TextOverflow.ELLIPSIS,
            ),
            content=ft.Container(
                width=680,
                height=480,
                bgcolor="#05080B",
                border_radius=12,
                clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                content=raw_image,
            ),
            actions=[
                ft.TextButton(
                    content=ft.Text("关闭", size=12, color=C.MUTED),
                    on_click=lambda e: self.page.pop_dialog(),
                )
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dialog)
        self._schedule_encoded_image(raw_image, data)

    def _read_attachment(self, attachment: Attachment) -> bytes | None:
        # 图片文件可能已被手动删除，读取失败时返回 None。
        try:
            return self.store.read_image(attachment)
        except OSError:
            return None

    # ------------------------------------------------------------------
    # 发送与流式生成
    # ------------------------------------------------------------------
    def _on_submit(self, e: Any = None) -> None:
        if self.generating:
            return
        self.page.run_task(self._send)

    def _send_clicked(self, e: Any = None) -> None:
        if self.generating:
            self._stop_generation()
        else:
            self.page.run_task(self._send)

    def _stop_generation(self) -> None:
        # 只设置停止标记，流式循环会在下一段内容处退出。
        if not self.generating:
            return
        self.stop_requested = True
        view = self.message_views.get(self.active_message_id or "")
        if view and view.status:
            view.status.value = "正在停止…"
            view.status.color = C.WARNING
            try:
                view.status.update()
            except Exception:
                pass

    async def _copy_text(self, text: str, message: str = "已复制到剪贴板。") -> None:
        # 剪贴板操作失败只提示，不影响其他功能。
        if not text:
            self._snack("没有可复制的内容。", tone="info")
            return
        try:
            await self.clipboard.set(text)
        except Exception as exc:  # noqa: BLE001 - clipboard is best-effort
            logger.warning("Could not copy to clipboard", exc_info=True)
            self._snack(f"复制失败：{exc}", tone="error")
            return
        self._snack(message, tone="success")

    async def _copy_message(self, message: Message) -> None:
        text = (message.content or "").strip() or (message.thinking or "").strip()
        if not text:
            self._snack("这条消息没有可复制的内容。", tone="info")
            return
        await self._copy_text(text)

    def _edit_message(self, message: Message) -> None:
        # 编辑后截断该消息之后的回复，再重新生成。
        if self.generating:
            self._snack("请先停止当前生成。", tone="warning")
            return
        conversation = self.current
        if conversation is None:
            return
        index = next(
            (
                position
                for position, item in enumerate(conversation.messages)
                if item.id == message.id
            ),
            None,
        )
        if index is None or message.role != "user":
            self._snack("只能编辑当前对话里的用户消息。", tone="info")
            return

        model = self.model_dropdown.value or conversation.model
        info = self._model_info(model)
        if not model:
            self._snack("请先选择一个模型。", tone="warning")
            return
        if message.attachments and not (info and info.supports_vision):
            self._snack(
                f"模型 {model} 不支持图片输入，请切换到视觉模型后再编辑。",
                tone="warning",
            )
            return

        field = ft.TextField(
            value=message.content,
            multiline=True,
            min_lines=3,
            max_lines=10,
            autofocus=True,
            width=520,
            text_size=13,
            color=C.TEXT,
            hint_text="输入新的消息内容…",
            hint_style=ft.TextStyle(color=C.MUTED_DIM, size=12),
            border=ft.OutlineInputBorder(
                side=ft.BorderSide(1, C.BORDER),
                border_radius=12,
            ),
            filled=True,
            fill_color=C.SURFACE_ALT,
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=10),
        )
        replies = max(0, len(conversation.messages) - index - 1)
        note = (
            f"保存后会删除这条消息之后的 {replies} 条回复，并重新请求模型。"
            if replies
            else "保存后会重新请求模型生成回答。"
        )
        if message.attachments:
            note += f" 这条消息附带的 {len(message.attachments)} 张图片会保留。"

        def save(event: Any) -> None:
            if self.generating:
                self._snack("请先停止当前生成。", tone="warning")
                return
            if self.current is None or self.current.id != conversation.id:
                self._snack("当前对话已切换，请重新操作。", tone="warning")
                return
            text = (field.value or "").strip()
            if not text and not message.attachments:
                field.error_text = "消息不能为空"
                field.update()
                return
            conversation.messages = conversation.messages[: index + 1]
            message.content = text
            message.created_at = time.time()
            conversation.model = model
            conversation.touch()
            self._persist_conversations()
            self.page.pop_dialog()
            self._set_follow_output(True)
            self._render_conversation()
            self._rebuild_sidebar()
            self._update_header()
            self._snack("消息已修改，正在重新生成…", tone="info")
            self.page.run_task(self._generate_reply, conversation, model)

        dialog = ft.AlertDialog(
            modal=True,
            bgcolor=C.SURFACE,
            shape=ft.RoundedRectangleBorder(radius=18),
            title=ft.Text(
                "编辑消息并重新发送",
                size=17,
                weight=ft.FontWeight.W_600,
                color=C.TEXT,
            ),
            content=ft.Column(
                controls=[field, ft.Text(note, size=10, color=C.MUTED_DIM)],
                spacing=10,
                tight=True,
            ),
            actions=[
                ft.TextButton(
                    content=ft.Text("取消", size=12, color=C.MUTED),
                    on_click=lambda event: self.page.pop_dialog(),
                ),
                ft.FilledButton(
                    content=ft.Text("保存并重发", size=12),
                    style=ft.ButtonStyle(
                        bgcolor=C.ACCENT,
                        color=C.BG,
                        padding=ft.Padding.symmetric(horizontal=16, vertical=10),
                        shape=ft.RoundedRectangleBorder(radius=10),
                    ),
                    on_click=save,
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dialog)

    def _regenerate_message(self, message: Message) -> None:
        # 只允许重新生成最后一条助手回答。
        if self.generating:
            self._snack("请先停止当前生成。", tone="warning")
            return
        conversation = self.current
        if (
            conversation is None
            or not conversation.messages
            or conversation.messages[-1].id != message.id
            or message.role != "assistant"
        ):
            self._snack("只能重新生成最后一条回答。", tone="info")
            return
        model = self.model_dropdown.value or conversation.model
        if not model:
            self._snack("请先选择一个模型。", tone="warning")
            return
        info = self._model_info(model)
        if (
            any(item.role == "user" and item.attachments for item in conversation.messages)
            and not (info and info.supports_vision)
        ):
            self._snack(
                f"模型 {model} 不支持图片输入，请切换到视觉模型后再重试。",
                tone="warning",
            )
            return

        conversation.messages.pop()
        self.message_views.pop(message.id, None)
        conversation.model = model
        conversation.touch()
        self._persist_conversations()
        self._set_follow_output(True)
        self._render_conversation()
        self._rebuild_sidebar()
        self._update_context_hint()
        self.page.run_task(self._generate_reply, conversation, model)

    async def _send(self, e: Any = None) -> None:
        # 发送流程：校验模型/图片 -> 保存附件 -> 追加用户消息 -> 生成回答。
        if self.generating:
            return
        text = (self.input.value or "").strip()
        images = list(self.pending_images)
        if not text and not images:
            self._snack("先输入消息或添加一张图片。", tone="info")
            self.page.run_task(self.input.focus)
            return
        model = self.model_dropdown.value
        if not model:
            self._snack("还没有可用的对话模型，请先安装并刷新模型。", tone="error")
            return
        info = self._model_info(model)
        if images and not (info and info.supports_vision):
            self._snack(
                f"模型 {model} 不支持图片输入，请切换视觉模型。", tone="warning"
            )
            return

        # 第一条消息发送时才真正创建会话。
        if self.current is None:
            self.current = Conversation(model=model)
            self.conversations.append(self.current)
        conversation = self.current
        conversation.model = model
        conversation.touch()

        try:
            attachments = [
                self.store.save_image(image.name, image.mime, image.data)
                for image in images
            ]
        except OSError as exc:
            logger.exception("Could not store images")
            self._snack(f"保存图片失败：{exc}", tone="error")
            return

        user_message = Message(
            role="user", content=text, attachments=attachments
        )
        conversation.messages.append(user_message)
        conversation.set_title_from_first_message()

        self.pending_images.clear()
        self._refresh_attachments()
        self.input.value = ""
        try:
            self.input.update()
        except Exception:
            pass

        self._add_message_ui(user_message)
        self._set_chat_visible(True)
        self._update_header()
        self._rebuild_sidebar()
        self._persist_conversations()

        await self._generate_reply(conversation, model)

    async def _generate_reply(
        self, conversation: Conversation, model: str
    ) -> None:
        # 先插入一个助手占位消息，流式内容会持续写进它。
        self._set_follow_output(True)
        assistant_message = Message(role="assistant", content="", model=model)
        conversation.messages.append(assistant_message)
        self.active_message_id = assistant_message.id
        self._add_message_ui(assistant_message)
        self._set_generating(True)

        try:
            await self._run_generation(conversation, assistant_message, model)
        finally:
            # 清理空占位消息、保存会话，并恢复可交互状态。
            self.active_message_id = None
            self.stop_requested = False
            conversation.messages = [
                message
                for message in conversation.messages
                if not (
                    message.role == "assistant"
                    and not message.has_body
                    and not message.error
                )
            ]
            view = self.message_views.get(assistant_message.id)
            if (
                all(item.id != assistant_message.id for item in conversation.messages)
                and view is not None
                and view.row in self.chat_list.controls
            ):
                self.chat_list.controls.remove(view.row)
                self.message_views.pop(assistant_message.id, None)
            conversation.touch()
            self._persist_conversations()
            self._rebuild_sidebar()
            self._update_header()
            self._set_generating(False)
            self._update_header()
            self._update_context_hint()
            try:
                self.chat_list.update()
            except Exception:
                pass
            if self.connected:
                self.page.run_task(self._refresh_runtime_info)

    async def _run_generation(
        self, conversation: Conversation, assistant: Message, model: str
    ) -> None:
        # 组装请求上下文；占位消息本身不能发给模型。
        view = self.message_views.get(assistant.id)
        history = [
            message for message in conversation.messages if message.id != assistant.id
        ]
        model_info = self._model_info(model)
        supports_thinking = bool(model_info and model_info.supports_thinking)
        think_enabled = supports_thinking and self.settings.deep_thinking
        num_gpu = self._gpu_layers(model_info)
        stopped = False
        last_flush = 0.0
        first_chunk_seen = False
        has_images = any(
            message.role == "user" and message.attachments for message in history
        )
        last_user_query = next(
            (
                message.content.strip()
                for message in reversed(history)
                if message.role == "user" and message.content.strip()
            ),
            "",
        )

        # 联网搜索放在流式生成之前；结果只作为本次请求的临时上下文。
        search_context = ""
        wants_search = bool(self.settings.web_search and last_user_query)
        if view and view.status:
            view.status.value = (
                "正在联网搜索…"
                if wants_search
                else ("正在处理图片并生成…" if has_images else "正在等待模型响应…")
            )
            view.status.color = C.ACCENT
            try:
                view.status.update()
            except Exception:
                pass
        if view and view.spinner:
            view.spinner.visible = True
            try:
                view.spinner.update()
            except Exception:
                pass

        if wants_search:
            search_failed = False
            try:
                results = await search_web(last_user_query)
            except WebSearchError as exc:
                search_failed = True
                results = []
                self._snack(str(exc), tone="warning")
            except Exception as exc:  # noqa: BLE001 - 搜索失败不应阻断回答
                logger.exception("Unexpected web search error")
                search_failed = True
                results = []
                self._snack(f"联网搜索失败：{exc}", tone="warning")
            if not results and not search_failed:
                self._snack("联网搜索没有找到结果，已改用本地模型回答。", tone="info")
            assistant.sources = results
            self._sync_sources_view(view, assistant)
            search_context = build_search_context(last_user_query, results)
            if view and view.status:
                view.status.value = (
                    "正在处理图片和搜索结果并生成…"
                    if has_images
                    else "正在等待模型响应…"
                )
                try:
                    view.status.update()
                except Exception:
                    pass

        payload = build_api_messages(
            history,
            system_prompt=self.settings.system_prompt,
            image_loader=self.store.read_image,
        )
        if search_context:
            # 保留用户自己的系统提示词，把搜索资料放在其后。
            system_index = (
                1 if payload and payload[0].get("role") == "system" else 0
            )
            payload.insert(
                system_index, {"role": "system", "content": search_context}
            )

        try:
            if view and view.markdown:
                view.markdown.value = "▍"
                view.markdown.update()

            async for chunk in self.client.chat_stream(
                model,
                payload,
                temperature=self.settings.temperature,
                think=think_enabled if supports_thinking else None,
                num_ctx=self.settings.context_size,
                num_gpu=num_gpu,
            ):
                if self.stop_requested:
                    stopped = True
                    break
                if not first_chunk_seen and (
                    chunk.content or (chunk.thinking and think_enabled)
                ):
                    first_chunk_seen = True
                    if view and view.status:
                        view.status.value = (
                            "深度思考中…"
                            if chunk.thinking and think_enabled
                            else "正在生成…"
                        )
                        view.status.color = C.ACCENT
                        try:
                            view.status.update()
                        except Exception:
                            pass
                if chunk.thinking and think_enabled:
                    assistant.thinking += chunk.thinking
                if chunk.content:
                    assistant.content += chunk.content
                # 保存统计用于速度和上下文占用显示。
                if chunk.eval_count:
                    assistant.eval_count = chunk.eval_count
                if chunk.eval_duration_ns:
                    assistant.eval_duration_ns = chunk.eval_duration_ns
                if chunk.prompt_eval_count:
                    assistant.prompt_eval_count = chunk.prompt_eval_count
                if chunk.prompt_eval_duration_ns:
                    assistant.prompt_eval_duration_ns = chunk.prompt_eval_duration_ns

                now = time.monotonic()
                if (
                    view
                    and (chunk.content or (chunk.thinking and think_enabled))
                    and now - last_flush >= FLUSH_INTERVAL
                ):
                    self._sync_streaming_view(view, assistant)
                    last_flush = now
                if chunk.done:
                    break

            if (
                not assistant.content.strip()
                and not assistant.thinking.strip()
                and not stopped
            ):
                assistant.error = True
                assistant.content = "模型没有返回内容，请检查模型是否正常运行。"
        except OllamaError as exc:
            assistant.error = True
            assistant.content = str(exc)
        except asyncio.CancelledError:
            stopped = True
            raise
        except Exception as exc:  # noqa: BLE001 - surface unexpected failures in the UI
            logger.exception("Unexpected generation error")
            assistant.error = True
            assistant.content = f"生成失败：{exc}"
        finally:
            if view:
                self._sync_final_view(view, assistant, stopped)

    def _sync_streaming_view(self, view: MessageView, message: Message) -> None:
        try:
            if view.markdown:
                view.markdown.value = message.content or "▍"
                view.markdown.visible = not message.error
                view.markdown.update()
            if view.thinking_md:
                view.thinking_md.value = message.thinking
                if message.thinking.strip() and view.thinking_box:
                    view.thinking_box.visible = True
                    view.thinking_box.update()
                view.thinking_md.update()
        except Exception:
            logger.debug("Could not flush streaming chunk", exc_info=True)

    def _sync_final_view(self, view: MessageView, message: Message, stopped: bool) -> None:
        # 生成结束后切换到最终视图；有代码块时替换成可复制的代码卡片。
        try:
            if view.markdown:
                view.markdown.value = message.content
                view.markdown.visible = not message.error
                view.markdown.update()
            if view.content_host:
                rich = None if message.error else self._rich_content(message.content)
                view.content_host.content = rich or view.markdown
                view.content_host.visible = not message.error
                view.content_host.update()
            if view.thinking_md:
                view.thinking_md.value = message.thinking
                if view.thinking_box:
                    view.thinking_box.visible = bool(message.thinking.strip())
                    view.thinking_box.update()
                view.thinking_md.update()
            if view.error_box and view.error_text:
                view.error_text.value = message.content if message.error else ""
                view.error_box.visible = message.error
                view.error_box.update()
            if view.actions:
                view.actions.visible = bool(
                    message.content.strip() or message.thinking.strip() or message.error
                )
                view.actions.update()
            if view.regenerate_button and message.error:
                view.regenerate_button.content = ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.REPLAY_ROUNDED, size=13, color=C.MUTED),
                        ft.Text("重试", size=10, color=C.MUTED),
                    ],
                    spacing=4,
                    tight=True,
                )
                try:
                    view.regenerate_button.update()
                except Exception:
                    pass
            self._refresh_message_actions()
            if view.spinner:
                view.spinner.visible = False
                view.spinner.update()
            if view.status:
                if stopped:
                    view.status.value = "已停止生成"
                    view.status.color = C.WARNING
                elif message.error:
                    view.status.value = ""
                else:
                    view.status.value = self._message_status_text(message)
                    view.status.color = C.MUTED_DIM
                view.status.update()
        except Exception:
            logger.exception("Could not finalize message view")

    def _add_message_ui(self, message: Message) -> None:
        view = self._build_message_view(message)
        self.message_views[message.id] = view
        self.chat_list.controls.append(view.row)
        try:
            self.chat_list.update()
        except Exception:
            logger.debug("Could not append message", exc_info=True)
        self._refresh_message_actions()

    def _refresh_message_actions(self) -> None:
        # 生成中隐藏编辑/重试按钮，完成后只显示最后一条回答的重生成。
        if not self.current or not self.current.messages:
            return
        last_message = self.current.messages[-1]
        for message in self.current.messages:
            view = self.message_views.get(message.id)
            if not view:
                continue
            if view.edit_button and message.role == "user":
                visible = not self.generating
                if view.edit_button.visible != visible:
                    view.edit_button.visible = visible
                    try:
                        view.edit_button.update()
                    except Exception:
                        logger.debug("Could not update edit action", exc_info=True)
            if view.regenerate_button:
                visible = (
                    message.id == last_message.id
                    and message.role == "assistant"
                    and not self.generating
                )
                if view.regenerate_button.visible != visible:
                    view.regenerate_button.visible = visible
                    try:
                        view.regenerate_button.update()
                    except Exception:
                        logger.debug(
                            "Could not update regenerate action", exc_info=True
                        )

    def _set_chat_visible(self, visible: bool) -> None:
        # 有消息时显示列表，否则显示空状态引导。
        self.chat_list.visible = visible
        self.empty_wrap.visible = not visible
        try:
            self.page.update(self.chat_list, self.empty_wrap)
        except Exception:
            pass

    def _set_generating(self, generating: bool) -> None:
        # 生成中禁用模型/图片/删除等操作，发送按钮变成停止。
        self.generating = generating
        self.new_chat_button.opacity = 0.55 if generating else 1.0
        self.model_dropdown.disabled = generating or not bool(self.chat_models)
        self.attach_button.disabled = generating
        self.delete_button.disabled = generating or self.current is None
        self.refresh_button.disabled = generating or self.refreshing
        if generating:
            self.send_button.content = ft.Icon(
                ft.Icons.STOP_ROUNDED, size=19, color=C.BG
            )
            self.send_button.style = ft.ButtonStyle(
                bgcolor=C.DANGER,
                color=C.BG,
                padding=0,
                shape=ft.RoundedRectangleBorder(radius=20),
                overlay_color=ft.Colors.with_opacity(0.12, C.BG),
            )
            self.send_button.tooltip = "停止生成"
        else:
            self.send_button.content = ft.Icon(
                ft.Icons.ARROW_UPWARD_ROUNDED, size=19, color=C.BG
            )
            self.send_button.style = ft.ButtonStyle(
                bgcolor=C.ACCENT,
                color=C.BG,
                padding=0,
                shape=ft.RoundedRectangleBorder(radius=20),
                overlay_color=ft.Colors.with_opacity(0.12, C.BG),
            )
            self.send_button.tooltip = "发送（Enter）"
        try:
            self.new_chat_button.update()
            self.model_dropdown.update()
            self.attach_button.update()
            self.delete_button.update()
            self.refresh_button.update()
            self.send_button.update()
        except Exception:
            logger.debug("Could not update generating state", exc_info=True)
        self._refresh_message_actions()
        self._update_think_control()

    # ------------------------------------------------------------------
    # 设置
    # ------------------------------------------------------------------
    def _open_settings(self, e: Any = None) -> None:
        # 设置对话框按“连接 / 对话 / 计算与图片 / 数据”分组折叠。
        url_field = ft.TextField(
            label="Ollama 地址",
            value=self.settings.base_url,
            hint_text="http://localhost:11434",
            prefix_icon=ft.Icons.LAN_OUTLINED,
            width=460,
            text_size=12,
            color=C.TEXT,
            label_style=ft.TextStyle(color=C.MUTED, size=11),
            border=ft.OutlineInputBorder(
                side=ft.BorderSide(1, C.BORDER),
                border_radius=12,
            ),
            filled=True,
            fill_color=C.SURFACE_ALT,
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=10),
        )
        prompt_field = ft.TextField(
            label="系统提示词",
            value=self.settings.system_prompt,
            hint_text="例如：你是一个简洁、准确的中文助手。",
            multiline=True,
            min_lines=3,
            max_lines=5,
            width=460,
            height=96,
            text_size=12,
            color=C.TEXT,
            label_style=ft.TextStyle(color=C.MUTED, size=11),
            hint_style=ft.TextStyle(color=C.MUTED_DIM, size=11),
            border=ft.OutlineInputBorder(
                side=ft.BorderSide(1, C.BORDER),
                border_radius=12,
            ),
            filled=True,
            fill_color=C.SURFACE_ALT,
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=10),
        )
        temperature_value = ft.Text(
            f"{self.settings.temperature:.2f}",
            size=12,
            weight=ft.FontWeight.W_600,
            color=C.ACCENT,
        )
        temperature_slider = ft.Slider(
            min=0,
            max=2,
            divisions=40,
            value=self.settings.temperature,
            active_color=C.ACCENT,
            inactive_color=C.BORDER,
            thumb_color=C.ACCENT,
            on_change=None,
        )

        context_sizes = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
        if self.settings.context_size not in context_sizes:
            context_sizes.append(self.settings.context_size)
            context_sizes.sort()
        context_dropdown = ft.Dropdown(
            label="上下文长度 (num_ctx)",
            value=str(self.settings.context_size),
            options=[
                ft.DropdownOption(key=str(size), text=f"{size} tokens")
                for size in context_sizes
            ],
            width=460,
            dense=True,
            filled=True,
            fill_color=C.SURFACE_ALT,
            bgcolor=C.SURFACE_ALT,
            border=ft.OutlineInputBorder(
                side=ft.BorderSide(1, C.BORDER),
                border_radius=12,
                gap_padding=2,
            ),
            text_size=12,
            color=C.TEXT,
            label_style=ft.TextStyle(color=C.MUTED, size=11),
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=8),
        )
        # 可选的自定义上下文；填写后优先于上面的预设。
        custom_context_field = ft.TextField(
            label="自定义上下文（可选）",
            value="",
            hint_text="例如 98304；留空则使用上方预设",
            width=460,
            text_size=12,
            color=C.TEXT,
            label_style=ft.TextStyle(color=C.MUTED, size=11),
            hint_style=ft.TextStyle(color=C.MUTED_DIM, size=10),
            border=ft.OutlineInputBorder(
                side=ft.BorderSide(1, C.BORDER),
                border_radius=12,
            ),
            filled=True,
            fill_color=C.SURFACE_ALT,
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9]*"),
        )

        compute_dropdown = ft.Dropdown(
            label="计算模式",
            value=self.settings.compute_mode,
            options=[
                ft.DropdownOption(key="auto", text="自动（由 Ollama 分配）"),
                ft.DropdownOption(key="cpu", text="纯 CPU（不加载到 GPU）"),
                ft.DropdownOption(key="hybrid", text="CPU + GPU 混合（可自定义层数）"),
                ft.DropdownOption(key="gpu", text="纯 GPU（全部层）"),
            ],
            width=460,
            dense=True,
            filled=True,
            fill_color=C.SURFACE_ALT,
            bgcolor=C.SURFACE_ALT,
            border=ft.OutlineInputBorder(
                side=ft.BorderSide(1, C.BORDER),
                border_radius=12,
                gap_padding=2,
            ),
            text_size=12,
            color=C.TEXT,
            label_style=ft.TextStyle(color=C.MUTED, size=11),
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=8),
        )

        selected_info = self._model_info(self.model_dropdown.value)
        selected_layers = selected_info.layer_count if selected_info else 0
        layers_hint = (
            f"0 = 自动一半；当前模型 {selected_layers} 层"
            if selected_layers
            else "0 = 自动一半；层数未知时默认 16 层"
        )
        hybrid_layers_field = ft.TextField(
            label="混合模式 GPU 层数",
            value=(
                str(self.settings.hybrid_gpu_layers)
                if self.settings.hybrid_gpu_layers
                else ""
            ),
            hint_text=layers_hint,
            width=460,
            text_size=12,
            color=C.TEXT,
            label_style=ft.TextStyle(color=C.MUTED, size=11),
            hint_style=ft.TextStyle(color=C.MUTED_DIM, size=10),
            border=ft.OutlineInputBorder(
                side=ft.BorderSide(1, C.BORDER),
                border_radius=12,
            ),
            filled=True,
            fill_color=C.SURFACE_ALT,
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9]*"),
            disabled=self.settings.compute_mode != "hybrid",
        )

        def on_compute_change(event: Any) -> None:
            mode = str(event.control.value or "auto")
            hybrid_layers_field.disabled = mode != "hybrid"
            hybrid_layers_field.opacity = 1.0 if mode == "hybrid" else 0.55
            try:
                hybrid_layers_field.update()
            except Exception:
                pass

        compute_dropdown.on_select = on_compute_change

        compress_switch = ft.Switch(
            value=self.settings.compress_images,
            active_color=C.ACCENT,
            active_track_color=C.ACCENT_SOFT,
            inactive_thumb_color=C.MUTED,
            inactive_track_color=C.SURFACE_ALT,
        )
        compress_row = ft.Container(
            padding=ft.Padding.symmetric(horizontal=2, vertical=2),
            content=ft.Row(
                controls=[
                    ft.Column(
                        controls=[
                            ft.Text("压缩图片", size=12, color=C.TEXT),
                            ft.Text(
                                "最长边压到 1280、转为 JPEG，可显著减少视觉 token；"
                                "关闭后发送原图。",
                                size=10,
                                color=C.MUTED_DIM,
                            ),
                        ],
                        spacing=2,
                        expand=True,
                    ),
                    compress_switch,
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

        web_search_switch = ft.Switch(
            value=self.settings.web_search,
            active_color=C.ACCENT,
            active_track_color=C.ACCENT_SOFT,
            inactive_thumb_color=C.MUTED,
            inactive_track_color=C.SURFACE_ALT,
        )
        web_search_row = ft.Container(
            padding=ft.Padding.symmetric(horizontal=2, vertical=2),
            content=ft.Row(
                controls=[
                    ft.Column(
                        controls=[
                            ft.Text("联网搜索", size=12, color=C.TEXT),
                            ft.Text(
                                "回答前使用 Bing 搜索网页，并把结果作为临时上下文；"
                                "搜索词会发送给 Bing。来源会显示在回答下方。",
                                size=10,
                                color=C.MUTED_DIM,
                            ),
                        ],
                        spacing=2,
                        expand=True,
                    ),
                    web_search_switch,
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

        def on_temperature_change(event: Any) -> None:
            temperature_value.value = f"{float(event.control.value or 0):.2f}"
            temperature_value.update()

        temperature_slider.on_change = on_temperature_change

        def save(event: Any) -> None:
            try:
                base_url = normalize_base_url(url_field.value or "")
            except ValueError:
                url_field.error_text = "请输入有效的 http:// 或 https:// 地址"
                url_field.update()
                return
            raw_layers = (hybrid_layers_field.value or "").strip()
            try:
                hybrid_layers = int(raw_layers) if raw_layers else 0
            except ValueError:
                hybrid_layers_field.error_text = "请输入 0 或正整数"
                hybrid_layers_field.update()
                return
            if hybrid_layers < 0 or hybrid_layers > 999:
                hybrid_layers_field.error_text = "层数需在 0–999 之间"
                hybrid_layers_field.update()
                return
            raw_context = (custom_context_field.value or "").strip()
            try:
                context_size = (
                    int(raw_context)
                    if raw_context
                    else int(context_dropdown.value or 8192)
                )
            except ValueError:
                custom_context_field.error_text = "请输入整数"
                custom_context_field.update()
                return
            if context_size < 2048 or context_size > 262144:
                custom_context_field.error_text = "范围需在 2048–262144"
                custom_context_field.update()
                return
            self.settings = Settings(
                base_url=base_url,
                system_prompt=(prompt_field.value or "").strip(),
                temperature=float(temperature_slider.value or 0.7),
                last_model=self.model_dropdown.value or self.settings.last_model,
                deep_thinking=self.settings.deep_thinking,
                context_size=context_size,
                compute_mode=str(compute_dropdown.value or "auto"),
                hybrid_gpu_layers=hybrid_layers,
                compress_images=bool(compress_switch.value),
                web_search=bool(web_search_switch.value),
            )
            self.client = OllamaClient(base_url)
            self.runtime_models = []
            self._save_settings()
            self._update_context_hint()
            self._update_model_meta()
            self._update_header_subtitle()
            self._update_web_search_button()
            self.page.pop_dialog()
            self._snack("设置已保存。", tone="success")
            self.page.run_task(self.refresh_models, notify=False)
            self.page.run_task(self._refresh_runtime_info)

        def section(
            title: str,
            icon: ft.IconData,
            controls: list[ft.Control],
            *,
            expanded: bool = False,
        ) -> ft.Control:
            return ft.ExpansionTile(
                title=ft.Text(
                    title,
                    size=12,
                    weight=ft.FontWeight.W_600,
                    color=C.TEXT_SOFT,
                ),
                leading=ft.Icon(icon, size=16, color=C.MUTED),
                controls=[
                    ft.Container(
                        padding=ft.Padding(left=10, top=0, right=10, bottom=10),
                        content=ft.Column(
                            controls=controls,
                            spacing=10,
                            tight=True,
                        ),
                    )
                ],
                expanded=expanded,
                maintain_state=True,
                dense=True,
                min_tile_height=42,
                tile_padding=ft.Padding.symmetric(horizontal=10, vertical=2),
                controls_padding=ft.Padding(left=0, top=0, right=0, bottom=0),
                bgcolor=C.SURFACE_ALT,
                collapsed_bgcolor=ft.Colors.TRANSPARENT,
                text_color=C.TEXT_SOFT,
                collapsed_text_color=C.TEXT_SOFT,
                icon_color=C.MUTED,
                collapsed_icon_color=C.MUTED_DIM,
                shape=ft.RoundedRectangleBorder(radius=10),
                collapsed_shape=ft.RoundedRectangleBorder(radius=10),
            )

        dialog = ft.AlertDialog(
            modal=True,
            scrollable=True,
            bgcolor=C.SURFACE,
            shape=ft.RoundedRectangleBorder(radius=18),
            title=ft.Text("设置", size=18, weight=ft.FontWeight.W_600, color=C.TEXT),
            content=ft.Container(
                width=520,
                content=ft.Column(
                    controls=[
                        ft.Text(
                            "点击分组展开或收起。",
                            size=10,
                            color=C.MUTED_DIM,
                        ),
                        section(
                            "连接",
                            ft.Icons.LAN_OUTLINED,
                            [
                                url_field,
                                ft.Text(
                                    "Ollama 默认监听 http://localhost:11434。",
                                    size=10,
                                    color=C.MUTED_DIM,
                                ),
                            ],
                            expanded=True,
                        ),
                        section(
                            "对话",
                            ft.Icons.CHAT_OUTLINED,
                            [
                                prompt_field,
                                ft.Row(
                                    controls=[
                                        ft.Text(
                                            "随机性 (temperature)",
                                            size=11,
                                            color=C.MUTED,
                                        ),
                                        ft.Container(expand=True),
                                        temperature_value,
                                    ],
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                                temperature_slider,
                                ft.Text(
                                    "数值越高回答越随机；代码与事实问答建议 0.2–0.7。",
                                    size=10,
                                    color=C.MUTED_DIM,
                                ),
                                context_dropdown,
                                custom_context_field,
                                ft.Text(
                                    "自定义值优先于上方预设，范围 2048–262144。"
                                    "上下文越大，KV Cache 内存近似线性增长；"
                                    "131072 以上建议配合混合或纯 CPU 模式。修改后对下一次请求生效。",
                                    size=10,
                                    color=C.MUTED_DIM,
                                ),
                                web_search_row,
                            ],
                        ),
                        section(
                            "计算与图片",
                            ft.Icons.MEMORY_ROUNDED,
                            [
                                compute_dropdown,
                                hybrid_layers_field,
                                ft.Text(
                                    "混合模式默认放约一半层，也可以指定具体层数；"
                                    "0 表示自动。纯 GPU 模式在显存不足时可能加载失败。"
                                    "修改后下一次请求生效。",
                                    size=10,
                                    color=C.MUTED_DIM,
                                ),
                                compress_row,
                            ],
                        ),
                        section(
                            "数据与关于",
                            ft.Icons.FOLDER_OUTLINED,
                            [
                                ft.Row(
                                    controls=[
                                        ft.Text(
                                            f"对话保存在 {self.store.data_dir}",
                                            size=10,
                                            color=C.MUTED_DIM,
                                            selectable=True,
                                            max_lines=2,
                                            overflow=ft.TextOverflow.ELLIPSIS,
                                            expand=True,
                                        ),
                                        ft.Text(
                                            f"v{APP_VERSION}",
                                            size=10,
                                            color=C.MUTED_DIM,
                                        ),
                                    ],
                                    spacing=6,
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                                ft.Text(
                                    "对话可导出为 Markdown；图片以文件名引用，不会嵌入文档。",
                                    size=10,
                                    color=C.MUTED_DIM,
                                ),
                            ],
                        ),
                    ],
                    spacing=8,
                    tight=True,
                ),
            ),
            actions=[
                ft.TextButton(
                    content=ft.Text("取消", size=12, color=C.MUTED),
                    on_click=lambda event: self.page.pop_dialog(),
                ),
                ft.FilledButton(
                    content=ft.Text("保存设置", size=12),
                    style=ft.ButtonStyle(
                        bgcolor=C.ACCENT,
                        color=C.BG,
                        padding=ft.Padding.symmetric(horizontal=16, vertical=10),
                        shape=ft.RoundedRectangleBorder(radius=10),
                    ),
                    on_click=save,
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dialog)

    # ------------------------------------------------------------------
    # 持久化与界面工具
    # ------------------------------------------------------------------
    def _persist_conversations(self) -> None:
        # 统一处理保存异常，避免单次磁盘错误打断聊天。
        try:
            self.store.save_conversations(self.conversations)
        except OSError:
            logger.exception("Could not save conversations")
            self._snack("无法保存对话，请检查磁盘权限。", tone="error")

    def _save_settings(self) -> None:
        # 设置保存失败只提示，不让应用崩溃。
        try:
            self.store.save_settings(self.settings)
        except OSError:
            logger.exception("Could not save settings")
            self._snack("无法保存设置，请检查磁盘权限。", tone="error")

    async def _copy_install_command(self, e: Any = None) -> None:
        # 一键复制视觉模型安装命令。
        try:
            await self.clipboard.set(INSTALL_VISION_CMD)
            self._snack("安装命令已复制到剪贴板。", tone="success")
        except Exception as exc:  # noqa: BLE001 - clipboard support varies
            logger.exception("Clipboard failed")
            self._snack(f"复制失败：{exc}", tone="error")

    def _snack(self, message: str, tone: str = "info") -> None:
        # 统一底部提示的颜色和图标。
        palette = {
            "info": (C.ACCENT, ft.Icons.INFO_OUTLINE),
            "success": (C.SUCCESS, ft.Icons.CHECK_CIRCLE_OUTLINE),
            "warning": (C.WARNING, ft.Icons.WARNING_AMBER_ROUNDED),
            "error": (C.DANGER, ft.Icons.ERROR_OUTLINE),
        }
        color, icon = palette.get(tone, palette["info"])
        snack = ft.SnackBar(
            content=ft.Row(
                controls=[
                    ft.Icon(icon, size=16, color=color),
                    ft.Text(message, size=12, color=C.TEXT, expand=True),
                ],
                spacing=9,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=C.SURFACE_ALT,
            behavior=ft.SnackBarBehavior.FLOATING,
            margin=ft.Margin(left=28, right=28, bottom=16),
            width=420,
            shape=ft.RoundedRectangleBorder(radius=12),
            duration=3200,
        )
        try:
            self.page.show_dialog(snack)
        except Exception:
            logger.debug("Could not show snackbar", exc_info=True)


