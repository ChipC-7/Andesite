"""核心逻辑测试，不需要真实 Ollama 服务或图形界面。"""

from __future__ import annotations

import base64
import json
import logging
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ollama_ui.client import (
    OllamaClient,
    OllamaError,
    RunningModel,
    build_api_messages,
    gpu_layers_for_mode,
    normalize_base_url,
)
from ollama_ui.markdown_utils import split_fenced_code
from ollama_ui.models import (
    Attachment,
    Conversation,
    Message,
    Settings,
)
from ollama_ui.storage import Store

logging.disable(logging.CRITICAL)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQAAAABJRU5ErkJggg=="
)


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    """用内存数据模拟 Ollama 的模型、聊天和 /api/ps 接口。"""

    protocol_version = "HTTP/1.1"
    models: list[dict[str, Any]] = [
        {
            "name": "fake-text:latest",
            "details": {"parameter_size": "3B", "quantization_level": "Q4_K_M"},
            "capabilities": ["completion"],
        },
        {
            "name": "fake-vision:latest",
            "details": {"parameter_size": "7B", "quantization_level": "Q4_K_M"},
            "capabilities": ["completion", "vision"],
        },
    ]
    chat_payload: dict[str, Any] = {}
    ps_payload: dict[str, Any] = {
        "models": [
            {
                "name": "fake-vision:latest",
                "size": 4_000_000_000,
                "size_vram": 3_500_000_000,
                "details": {
                    "parameter_size": "7B",
                    "quantization_level": "Q4_K_M",
                },
            }
        ]
    }

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/version":
            self._write_json(200, {"version": "0.30.10-fake"})
        elif self.path == "/api/tags":
            self._write_json(200, {"models": self.models})
        elif self.path == "/api/ps":
            self._write_json(200, self.ps_payload)
        else:
            self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(size) or b"{}")
        if self.path == "/api/show":
            model = next(
                (m for m in self.models if m["name"] == payload.get("model")), None
            )
            capabilities = list(model["capabilities"]) if model else []
            # 模拟 /api/tags 缺少 thinking、只能从 /api/show 获取的情况。
            if model and model["name"] == "fake-vision:latest":
                capabilities.append("thinking")
            self._write_json(
                200,
                {
                    "capabilities": capabilities,
                    "details": model["details"] if model else {},
                    "model_info": {"fake.block_count": 4} if model else {},
                },
            )
            return
        if self.path == "/api/chat":
            type(self).chat_payload = payload
            model = payload.get("model")
            if model not in {m["name"] for m in self.models}:
                self._write_json(404, {"error": f"model '{model}' not found"})
                return
            last_user = next(
                (
                    message
                    for message in reversed(payload.get("messages", []))
                    if message.get("role") == "user"
                ),
                {},
            )
            image_count = len(last_user.get("images") or [])
            # 依次返回思考、正文和最终统计三段流式数据。
            chunks = [
                {"message": {"content": "", "thinking": "思考"}, "done": False},
                {
                    "message": {
                        "content": f"收到 {image_count} 张图片。\n\n```python\nprint(1)\n```"
                    },
                    "done": False,
                },
                {
                    "message": {"content": ""},
                    "done": True,
                    "eval_count": 42,
                    "eval_duration": 1_000_000_000,
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 2_000_000_000,
                },
            ]
            body = b"".join((json.dumps(chunk) + "\n").encode() for chunk in chunks)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._write_json(404, {"error": "not found"})


class StoreTests(unittest.TestCase):
    """本地 JSON 和图片持久化。"""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = Store(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_settings_roundtrip(self) -> None:
        settings = Settings(
            base_url="http://localhost:11434",
            system_prompt="你是助手",
            temperature=0.3,
            last_model="fake-vision:latest",
            deep_thinking=False,
            context_size=16384,
            compute_mode="hybrid",
            hybrid_gpu_layers=12,
            compress_images=False,
        )
        self.store.save_settings(settings)
        loaded = self.store.load_settings()
        self.assertEqual(loaded.base_url, settings.base_url)
        self.assertEqual(loaded.system_prompt, settings.system_prompt)
        self.assertAlmostEqual(loaded.temperature, 0.3)
        self.assertEqual(loaded.last_model, settings.last_model)
        self.assertFalse(loaded.deep_thinking)
        self.assertEqual(loaded.context_size, 16384)
        self.assertEqual(loaded.compute_mode, "hybrid")
        self.assertEqual(loaded.hybrid_gpu_layers, 12)
        self.assertFalse(loaded.compress_images)

    def test_settings_clamps_temperature(self) -> None:
        self.store.save_settings(Settings(temperature=9.0))
        loaded = self.store.load_settings()
        self.assertEqual(loaded.temperature, 2.0)

    def test_settings_clamps_hybrid_layers(self) -> None:
        self.store.save_settings(
            Settings.from_dict({"hybrid_gpu_layers": -3})
        )
        self.assertEqual(self.store.load_settings().hybrid_gpu_layers, 0)
        self.store.save_settings(
            Settings.from_dict({"hybrid_gpu_layers": 2000})
        )
        self.assertEqual(self.store.load_settings().hybrid_gpu_layers, 999)

    def test_settings_clamps_context_size(self) -> None:
        self.store.save_settings(
            Settings.from_dict({"context_size": 524288})
        )
        self.assertEqual(self.store.load_settings().context_size, 262144)
        self.store.save_settings(
            Settings.from_dict({"context_size": 1024})
        )
        self.assertEqual(self.store.load_settings().context_size, 2048)

    def test_settings_keeps_custom_context_size(self) -> None:
        self.store.save_settings(
            Settings.from_dict({"context_size": 98304})
        )
        self.assertEqual(self.store.load_settings().context_size, 98304)

    def test_conversation_roundtrip_and_drops_empty_assistant(self) -> None:
        attachment = self.store.save_image("test.png", "image/png", PNG_BYTES)
        conversation = Conversation(
            title="图片对话",
            model="fake-vision:latest",
            messages=[
                Message(
                    role="user",
                    content="看图",
                    attachments=[attachment],
                ),
                Message(role="assistant", content="", model="fake-vision:latest"),
                Message(
                    role="assistant",
                    content="看到了",
                    model="fake-vision:latest",
                    eval_count=10,
                    eval_duration_ns=500_000_000,
                    prompt_eval_count=7,
                    prompt_eval_duration_ns=100_000_000,
                ),
            ],
        )
        conversations = [conversation]
        self.store.save_conversations(conversations)
        loaded = self.store.load_conversations()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].title, "图片对话")
        self.assertEqual(len(loaded[0].messages), 2)
        self.assertEqual(loaded[0].messages[0].attachments[0].name, "test.png")
        self.assertEqual(self.store.read_image(attachment), PNG_BYTES)

    def test_image_lifecycle(self) -> None:
        attachment = self.store.save_image(
            "screen.webp", "image/webp", b"webp-data"
        )
        self.assertEqual(Path(attachment.path).suffix, ".webp")
        self.assertEqual(self.store.read_image(attachment), b"webp-data")
        self.store.delete_images([attachment])
        self.assertFalse((self.store.media_dir / attachment.path).exists())

    def test_invalid_json_does_not_crash(self) -> None:
        self.store.conversations_path.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.store.load_conversations(), [])


class ModelTests(unittest.TestCase):
    """会话模型、搜索、置顶和 Markdown 导出。"""

    def test_conversation_title_and_preview(self) -> None:
        conversation = Conversation()
        conversation.messages.append(
            Message(role="user", content="  请解释一下这个报错  ")
        )
        conversation.set_title_from_first_message()
        self.assertEqual(conversation.title, "请解释一下这个报错")
        self.assertEqual(conversation.preview, "请解释一下这个报错")

    def test_collapse_long_title(self) -> None:
        conversation = Conversation(
            messages=[Message(role="user", content="a" * 80)]
        )
        conversation.set_title_from_first_message()
        self.assertTrue(conversation.title.endswith("…"))
        self.assertLessEqual(len(conversation.title), 32)

    def test_conversation_search_matches_content_and_model(self) -> None:
        conversation = Conversation(
            title="报错分析",
            model="qwen3.5:4b",
            messages=[
                Message(role="user", content="这个异常怎么处理？"),
                Message(role="assistant", content="检查一下 OOM 设置", thinking="先看日志"),
            ],
        )
        self.assertTrue(conversation.matches(""))
        self.assertTrue(conversation.matches("报错"))
        self.assertTrue(conversation.matches("异常"))
        self.assertTrue(conversation.matches("oom"))
        self.assertTrue(conversation.matches("日志"))
        self.assertTrue(conversation.matches("qwen"))
        self.assertFalse(conversation.matches("图片"))

    def test_pinned_conversation_roundtrip(self) -> None:
        conversation = Conversation(title="置顶对话", pinned=True)
        loaded = Conversation.from_dict(conversation.to_dict())
        self.assertTrue(loaded.pinned)
        self.assertEqual(loaded.title, "置顶对话")

    def test_markdown_export_includes_messages_and_images(self) -> None:
        attachment = Attachment(name="screenshot.png", mime="image/png", path="x")
        conversation = Conversation(
            title="导出测试",
            model="qwen3.5:4b",
            messages=[
                Message(role="user", content="看看这张图", attachments=[attachment]),
                Message(
                    role="assistant",
                    content="这是代码：\n\n```python\nprint(1)\n```",
                    thinking="先识别图片",
                    model="qwen3.5:4b",
                    eval_count=20,
                    eval_duration_ns=1_000_000_000,
                ),
            ],
        )
        markdown = conversation.to_markdown()
        self.assertIn("# 导出测试", markdown)
        self.assertIn("## 你", markdown)
        self.assertIn("screenshot.png", markdown)
        self.assertIn("## 助手", markdown)
        self.assertIn("print(1)", markdown)
        self.assertIn("<summary>思考过程</summary>", markdown)
        self.assertIn("20 tokens", markdown)

    def test_base_url_normalization(self) -> None:
        self.assertEqual(
            normalize_base_url("localhost:11434"), "http://localhost:11434"
        )
        self.assertEqual(
            normalize_base_url("http://localhost:11434/"),
            "http://localhost:11434",
        )
        with self.assertRaises(ValueError):
            normalize_base_url("ftp://localhost:11434")


class ApiMessageTests(unittest.TestCase):
    """构造 Ollama 请求时的消息和图片策略。"""

    def test_system_prompt_and_images(self) -> None:
        attachment = Attachment(
            name="a.png", mime="image/png", path="media/a.png", size=3
        )
        messages = [
            Message(role="user", content="看图", attachments=[attachment]),
            Message(role="assistant", content=""),
            Message(role="assistant", content="回答"),
        ]
        payload = build_api_messages(
            messages,
            system_prompt="你是助手",
            image_loader=lambda _: b"abc",
        )
        self.assertEqual(payload[0], {"role": "system", "content": "你是助手"})
        self.assertEqual(payload[1]["role"], "user")
        self.assertEqual(
            payload[1]["images"], [base64.b64encode(b"abc").decode()]
        )
        self.assertEqual([m["role"] for m in payload], ["system", "user", "assistant"])

    def test_skips_empty_messages(self) -> None:
        payload = build_api_messages([Message(role="assistant", content="")])
        self.assertEqual(payload, [])

    def test_skips_failed_responses(self) -> None:
        payload = build_api_messages(
            [
                Message(role="user", content="hi"),
                Message(role="assistant", content="连接失败", error=True),
            ]
        )
        self.assertEqual([message["role"] for message in payload], ["user"])

    def test_only_latest_image_message_is_sent(self) -> None:
        old_image = Attachment(name="old.png", mime="image/png", path="old", size=3)
        new_image = Attachment(name="new.png", mime="image/png", path="new", size=3)
        messages = [
            Message(role="user", content="看图一", attachments=[old_image]),
            Message(role="assistant", content="看到了"),
            Message(role="user", content="看图二", attachments=[new_image]),
        ]
        payload = build_api_messages(
            messages,
            image_loader=lambda attachment: attachment.path.encode(),
        )
        self.assertNotIn("images", payload[0])
        self.assertEqual(
            payload[2]["images"], [base64.b64encode(b"new").decode()]
        )

    def test_keeps_previous_images_for_text_followup(self) -> None:
        image = Attachment(name="old.png", mime="image/png", path="old", size=3)
        messages = [
            Message(role="user", content="看图", attachments=[image]),
            Message(role="assistant", content="看到了"),
            Message(role="user", content="再解释一下"),
        ]
        payload = build_api_messages(
            messages, image_loader=lambda attachment: attachment.path.encode()
        )
        self.assertEqual(
            payload[0]["images"], [base64.b64encode(b"old").decode()]
        )
        self.assertNotIn("images", payload[2])


class ClientTests(unittest.IsolatedAsyncioTestCase):
    """通过本地假服务器测试 Ollama 客户端。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address
        cls.base_url = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    async def test_list_models_and_version(self) -> None:
        client = OllamaClient(self.base_url)
        models = await client.list_models()
        self.assertEqual([m.name for m in models], ["fake-text:latest", "fake-vision:latest"])
        self.assertFalse(models[0].supports_vision)
        self.assertTrue(models[1].supports_vision)
        self.assertFalse(models[0].supports_thinking)
        self.assertTrue(models[1].supports_thinking)
        self.assertEqual(models[1].layer_count, 4)
        self.assertEqual(models[0].parameter_size, "3B")
        self.assertEqual(await client.get_version(), "0.30.10-fake")

    async def test_running_models_report_vram(self) -> None:
        client = OllamaClient(self.base_url)
        running = await client.running_models()
        self.assertEqual(len(running), 1)
        self.assertEqual(running[0].name, "fake-vision:latest")
        self.assertEqual(running[0].parameter_size, "7B")
        self.assertAlmostEqual(running[0].vram_share, 0.875)

    async def test_stream_chat(self) -> None:
        client = OllamaClient(self.base_url)
        image = base64.b64encode(PNG_BYTES).decode()
        chunks = []
        async for chunk in client.chat_stream(
            "fake-vision:latest",
            [
                {
                    "role": "user",
                    "content": "看图",
                    "images": [image],
                }
            ],
            temperature=0.4,
            think=True,
            num_ctx=8192,
            num_gpu=2,
        ):
            chunks.append(chunk)
        content = "".join(chunk.content for chunk in chunks)
        thinking = "".join(chunk.thinking for chunk in chunks)
        self.assertIn("收到 1 张图片", content)
        self.assertEqual(thinking, "思考")
        self.assertTrue(chunks[-1].done)
        self.assertEqual(chunks[-1].eval_count, 42)
        payload = _FakeOllamaHandler.chat_payload
        self.assertEqual(payload["options"]["temperature"], 0.4)
        self.assertIs(payload["think"], True)
        self.assertEqual(payload["options"]["num_ctx"], 8192)
        self.assertEqual(payload["options"]["num_gpu"], 2)
        self.assertEqual(chunks[-1].prompt_eval_count, 120)
        self.assertEqual(chunks[-1].prompt_eval_duration_ns, 2_000_000_000)
        self.assertEqual(payload["messages"][0]["images"], [image])

    async def test_missing_model_has_friendly_error(self) -> None:
        client = OllamaClient(self.base_url)
        with self.assertRaises(OllamaError) as context:
            async for _ in client.chat_stream(
                "missing:latest", [{"role": "user", "content": "hi"}]
            ):
                pass
        self.assertIn("不存在", str(context.exception))

    async def test_offline_error(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        client = OllamaClient(f"http://127.0.0.1:{port}")
        with self.assertRaises(OllamaError) as context:
            await client.list_models()
        self.assertIn("无法连接", str(context.exception))



class ErrorFormattingTests(unittest.TestCase):
    """原始错误到中文提示的转换。"""

    def test_context_overflow_is_friendly(self) -> None:
        body = json.dumps(
            {
                "error": {
                    "code": 400,
                    "message": "request (6206 tokens) exceeds the available context size (4096 tokens), try increasing it",
                    "type": "exceed_context_size_error",
                    "n_prompt_tokens": 6206,
                    "n_ctx": 4096,
                }
            }
        ).encode()
        message = OllamaClient._error_from_body(body, 400, "qwen3.5:4b")
        self.assertIn("上下文长度不足", message)
        self.assertIn("6206", message)
        self.assertIn("4096", message)
        self.assertIn("num_ctx", message)

    def test_memory_error_is_friendly(self) -> None:
        body = json.dumps(
            {"error": {"code": 500, "message": "failed to allocate buffer"}}
        ).encode()
        message = OllamaClient._error_from_body(body, 500, "qwen3.5:4b")
        self.assertIn("显存或内存不足", message)
        self.assertIn("计算模式", message)


class ImageCompressionTests(unittest.TestCase):
    """图片压缩行为。"""

    def test_large_image_is_downscaled_to_jpeg(self) -> None:
        from io import BytesIO
        from PIL import Image

        from ollama_ui.images import compress_image

        source = Image.new("RGB", (2000, 1000), (30, 120, 200))
        buffer = BytesIO()
        source.save(buffer, format="PNG")
        data = buffer.getvalue()
        result, mime, name = compress_image(data, "large.png", "image/png")
        self.assertEqual(mime, "image/jpeg")
        self.assertTrue(name.endswith(".jpg"))
        self.assertLess(len(result), len(data))
        with Image.open(BytesIO(result)) as prepared:
            self.assertLessEqual(max(prepared.size), 1280)
            self.assertEqual(prepared.format, "JPEG")


class MarkdownSplittingTests(unittest.TestCase):
    """Markdown 代码块拆分。"""

    def test_splits_text_and_code_blocks(self) -> None:
        text = "说明：\n\n```python\nprint(1)\n```\n\n结束。"
        segments = split_fenced_code(text)
        self.assertEqual([segment.kind for segment in segments], ["text", "code", "text"])
        self.assertEqual(segments[1].language, "python")
        self.assertEqual(segments[1].text, "print(1)")
        self.assertIn("说明", segments[0].text)
        self.assertIn("结束", segments[2].text)

    def test_multiple_code_blocks_and_empty_language(self) -> None:
        text = "```js\nconst a = 1;\n```\n\n```\nplain\n```"
        segments = split_fenced_code(text)
        code = [segment for segment in segments if segment.kind == "code"]
        self.assertEqual(len(code), 2)
        self.assertEqual(code[0].language, "js")
        self.assertEqual(code[1].language, "")
        self.assertEqual(code[1].text, "plain")

    def test_unclosed_fence_stays_as_text(self) -> None:
        segments = split_fenced_code("```python\nprint(1)\n")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].kind, "text")
        self.assertIn("```python", segments[0].text)


class ComputeModeTests(unittest.TestCase):
    """计算模式和 GPU 层数映射。"""

    def test_gpu_layer_mapping(self) -> None:
        self.assertEqual(gpu_layers_for_mode("cpu", 32), 0)
        self.assertEqual(gpu_layers_for_mode("gpu", 32), 999)
        self.assertEqual(gpu_layers_for_mode("hybrid", 32), 16)
        self.assertEqual(gpu_layers_for_mode("hybrid", 0), 16)
        self.assertIsNone(gpu_layers_for_mode("auto", 32))

    def test_custom_hybrid_layers(self) -> None:
        self.assertEqual(gpu_layers_for_mode("hybrid", 32, 12), 12)
        self.assertEqual(gpu_layers_for_mode("hybrid", 32, 99), 32)
        self.assertEqual(gpu_layers_for_mode("hybrid", 0, 7), 7)
        self.assertEqual(gpu_layers_for_mode("hybrid", 32, 0), 16)

    def test_running_model_vram_share_is_clamped(self) -> None:
        running = RunningModel(name="m", size=100, size_vram=150)
        self.assertEqual(running.vram_share, 1.0)
        self.assertEqual(RunningModel(name="m").vram_share, 0.0)


if __name__ == "__main__":
    unittest.main()
