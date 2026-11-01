"""桌面界面的颜色和 Markdown 样式。"""

from __future__ import annotations

import flet as ft


class C:
    """颜色令牌：深色蓝黑底 + 薄荷绿强调色。"""

    # 背景层次：窗口、渐变顶部、侧栏、卡片、次级卡片。
    BG = "#0B0F14"
    BG_TOP = "#0E151D"
    SIDEBAR = "#0C1118"
    SURFACE = "#131B24"
    SURFACE_ALT = "#17212C"
    BORDER = "#23303D"
    BORDER_SOFT = "#18222C"

    # 文字层次：主文字、次文字、弱化文字、小号说明文字。
    TEXT = "#EAF0F7"
    TEXT_SOFT = "#C7D1DE"
    MUTED = "#8492A6"
    MUTED_DIM = "#7C8CA1"

    # 品牌强调色及其浅色底、描边。
    ACCENT = "#5EEAD4"
    ACCENT_HOVER = "#7FF0DE"
    ACCENT_TEXT = "#A7F3E7"
    ACCENT_SOFT = "#12302C"
    ACCENT_BORDER = "#1E4F47"

    # 用户消息气泡。
    USER_BG = "#12332F"
    USER_BORDER = "#1D4A43"

    # 错误状态。
    DANGER = "#F87171"
    DANGER_SOFT = "#2A171A"
    DANGER_BORDER = "#4A252A"

    # 警告状态。
    WARNING = "#FBBF24"
    WARNING_SOFT = "#2A2313"
    WARNING_BORDER = "#4A3A1A"

    SUCCESS = "#34D399"
    CODE_BG = "#0D1520"


def build_theme() -> ft.Theme:
    # 把颜色令牌映射到 Material 主题，保证按钮、弹窗等控件风格一致。
    return ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=C.ACCENT,
            on_primary=C.BG,
            primary_container=C.ACCENT_SOFT,
            on_primary_container=C.ACCENT_TEXT,
            secondary=C.ACCENT,
            on_secondary=C.BG,
            surface=C.SURFACE,
            on_surface=C.TEXT,
            surface_container=C.SURFACE,
            surface_container_high=C.SURFACE_ALT,
            on_surface_variant=C.MUTED,
            outline=C.BORDER,
            outline_variant=C.BORDER_SOFT,
            error=C.DANGER,
            on_error=C.BG,
            error_container=C.DANGER_SOFT,
            on_error_container=C.DANGER,
        ),
        scaffold_bgcolor=C.BG,
        card_bgcolor=C.SURFACE,
        divider_color=C.BORDER_SOFT,
        hint_color=C.MUTED_DIM,
        splash_color=ft.Colors.with_opacity(0.16, C.ACCENT),
        highlight_color=ft.Colors.with_opacity(0.08, C.ACCENT),
        hover_color=ft.Colors.with_opacity(0.05, C.TEXT),
    )


def markdown_style() -> ft.MarkdownStyleSheet:
    # 正文、标题、强调和链接的字体层级。
    return ft.MarkdownStyleSheet(
        p_text_style=ft.TextStyle(color=C.TEXT, size=15, height=1.65),
        h1_text_style=ft.TextStyle(
            color=C.TEXT, size=24, weight=ft.FontWeight.W_700, height=1.3
        ),
        h2_text_style=ft.TextStyle(
            color=C.TEXT, size=20, weight=ft.FontWeight.W_700, height=1.35
        ),
        h3_text_style=ft.TextStyle(
            color=C.TEXT, size=17, weight=ft.FontWeight.W_600, height=1.4
        ),
        h4_text_style=ft.TextStyle(
            color=C.TEXT, size=15, weight=ft.FontWeight.W_600
        ),
        h5_text_style=ft.TextStyle(
            color=C.TEXT_SOFT, size=15, weight=ft.FontWeight.W_600
        ),
        h6_text_style=ft.TextStyle(
            color=C.MUTED, size=14, weight=ft.FontWeight.W_600
        ),
        strong_text_style=ft.TextStyle(color=C.TEXT, weight=ft.FontWeight.W_700),
        em_text_style=ft.TextStyle(color=C.TEXT_SOFT, italic=True),
        a_text_style=ft.TextStyle(color=C.ACCENT, decoration=ft.TextDecoration.UNDERLINE),
        # 行内代码和代码块使用等宽字体与深色底。
        code_text_style=ft.TextStyle(
            color=C.ACCENT_TEXT, size=13, font_family="monospace"
        ),
        blockquote_text_style=ft.TextStyle(color=C.MUTED, italic=True, height=1.55),
        blockquote_decoration=ft.BoxDecoration(
            bgcolor=C.SURFACE_ALT,
            border=ft.Border(left=ft.BorderSide(2, C.ACCENT_BORDER)),
        ),
        codeblock_decoration=ft.BoxDecoration(
            bgcolor=C.CODE_BG,
            border=ft.Border.all(1, C.BORDER_SOFT),
            border_radius=10,
        ),
        codeblock_padding=12,
        blockquote_padding=12,
        block_spacing=10,
        list_indent=20,
        list_bullet_text_style=ft.TextStyle(color=C.ACCENT),
        horizontal_rule_decoration=ft.BoxDecoration(bgcolor=C.BORDER),
        table_cells_padding=6,
    )
