"""Andesite 桌面应用入口。"""

from __future__ import annotations

import logging

import flet as ft

from andesite.app import AndesiteApp


def main(page: ft.Page) -> None:
    # 统一日志格式，并压低依赖库的噪声日志。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    )
    for noisy in ("httpx", "flet_web", "flet"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    AndesiteApp(page)


if __name__ == "__main__":
    ft.run(main, name="Andesite", assets_dir=None)
