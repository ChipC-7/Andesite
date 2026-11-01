"""轻量的 Ollama HTTP 客户端。

只实现界面需要的接口：``/api/tags``、``/api/show``、``/api/version``、
``/api/ps`` 和流式 ``/api/chat``。依赖很少，便于直接配合本机 Ollama 使用。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# 旧版 Ollama 不返回 capabilities 时，用模型名做视觉能力兜底判断。
_VISION_NAME_HINTS = (
    "llava",
    "vision",
    "moondream",
    "minicpm-v",
    "qwen2.5vl",
    "qwen2-vl",
    "qwen3-vl",
    "gemma3",
)


class OllamaError(RuntimeError):
    """可以直接展示给用户的 Ollama 错误。"""


def normalize_base_url(value: str) -> str:
    """整理地址；允许直接输入 ``localhost:11434``。"""
    url = (value or "").strip().rstrip("/")
    if not url:
        url = "http://localhost:11434"
    if "://" not in url:
        url = f"http://{url}"
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Ollama address must be a valid http:// or https:// URL")
    return url


@dataclass(frozen=True)
class ModelInfo:
    name: str
    capabilities: tuple[str, ...] = ()
    parameter_size: str = ""
    quantization: str = ""
    family: str = ""
    size_bytes: int = 0
    layer_count: int = 0

    @property
    def supports_vision(self) -> bool:
        if "vision" in self.capabilities:
            return True
        if self.capabilities:
            return False
        # 旧版没有 capabilities 时，按名称保守推断。
        lowered = self.name.lower()
        return any(hint in lowered for hint in _VISION_NAME_HINTS)

    @property
    def is_chat_model(self) -> bool:
        caps = set(self.capabilities)
        if not caps:
            return True
        # 只有 embedding 能力的模型不参与聊天。
        return bool(caps - {"embedding"})

    @property
    def supports_thinking(self) -> bool:
        return "thinking" in self.capabilities

    @property
    def subtitle(self) -> str:
        bits = [bit for bit in (self.parameter_size, self.quantization) if bit]
        return " · ".join(bits)


def gpu_layers_for_mode(
    mode: str, layer_count: int = 0, custom_layers: int = 0
) -> int | None:
    """把计算模式转换成 Ollama 的 ``num_gpu``。

    返回 ``None`` 表示不传该参数，让 Ollama 自动分配。混合模式下
    ``custom_layers`` 为正数时覆盖自动一半，超过模型层数会被收敛。
    """
    if mode == "cpu":
        return 0
    if mode == "gpu":
        # 999 由 Ollama 自动收敛到模型实际层数。
        return 999
    if mode == "hybrid":
        if custom_layers > 0:
            layers = max(1, int(custom_layers))
            if layer_count > 0:
                layers = min(layers, layer_count)
            return layers
        if layer_count > 0:
            return max(1, layer_count // 2)
        return 16
    return None


@dataclass
class ChatChunk:
    """流式返回的一小段内容或最终统计。"""

    content: str = ""
    thinking: str = ""
    done: bool = False
    eval_count: int = 0
    eval_duration_ns: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0


@dataclass(frozen=True)
class RunningModel:
    """``/api/ps`` 返回的已加载模型。"""

    name: str
    size: int = 0
    size_vram: int = 0
    parameter_size: str = ""
    quantization: str = ""
    expires_at: str = ""

    @property
    def vram_share(self) -> float:
        # 显存占比，限制在 0~1。
        if self.size <= 0:
            return 0.0
        return min(1.0, max(0.0, self.size_vram / self.size))


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = normalize_base_url(base_url)

    # ------------------------------------------------------------------
    # 模型发现
    # ------------------------------------------------------------------
    async def list_models(self) -> list[ModelInfo]:
        # 延迟导入 httpx，减少应用启动时的 Python 导入时间。
        import httpx

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                payload = response.json()
                models = payload.get("models") or []

                infos: list[ModelInfo] = []
                for item in models:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or item.get("model") or "")
                    if not name:
                        continue
                    details = item.get("details") or {}
                    capabilities = tuple(
                        str(cap) for cap in (item.get("capabilities") or [])
                    )
                    infos.append(
                        ModelInfo(
                            name=name,
                            capabilities=capabilities,
                            parameter_size=str(
                                details.get("parameter_size") or ""
                            ),
                            quantization=str(
                                details.get("quantization_level") or ""
                            ),
                            family=str(details.get("family") or ""),
                            size_bytes=int(item.get("size") or 0),
                            layer_count=0,
                        )
                    )

                # /api/tags 可能缺少 thinking 等新能力，逐个用 /api/show 补齐。
                if infos:
                    infos = list(
                        await asyncio.gather(
                            *(self._enrich_model(client, info) for info in infos)
                        )
                    )

                return sorted(infos, key=lambda info: info.name.lower())
        except httpx.ConnectError as exc:
            raise OllamaError(
                "无法连接到 Ollama。请确认服务正在运行"
                f"（{self.base_url}），可执行 `ollama serve` 启动。"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaError(
                f"连接 Ollama 超时（{self.base_url}）。请检查地址或服务状态。"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaError(
                f"Ollama 返回 HTTP {exc.response.status_code}。"
                "请检查服务版本或访问地址。"
            ) from exc
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise OllamaError(f"读取 Ollama 模型列表失败：{exc}") from exc

    async def _enrich_model(
        self, client: httpx.AsyncClient, info: ModelInfo
    ) -> ModelInfo:
        import httpx

        try:
            # 单个模型详情卡住时不要拖慢整个启动，超时后回退 tags 数据。
            response = await client.post(
                f"{self.base_url}/api/show",
                json={"model": info.name},
                timeout=5.0,
            )
            response.raise_for_status()
            payload = response.json()
            capabilities = tuple(
                str(cap) for cap in (payload.get("capabilities") or [])
            )
            details = payload.get("details") or {}
            model_info = payload.get("model_info") or {}
            # 层数用于混合模式计算 num_gpu。
            layer_count = info.layer_count
            for key, value in model_info.items():
                if (
                    str(key).endswith(".block_count")
                    and isinstance(value, int)
                    and value > 0
                ):
                    layer_count = value
                    break
            merged = tuple(dict.fromkeys((*info.capabilities, *capabilities)))
            return ModelInfo(
                name=info.name,
                capabilities=merged,
                parameter_size=info.parameter_size
                or str(details.get("parameter_size") or ""),
                quantization=info.quantization
                or str(details.get("quantization_level") or ""),
                family=info.family or str(details.get("family") or ""),
                size_bytes=info.size_bytes,
                layer_count=layer_count,
            )
        except (httpx.HTTPError, ValueError, TypeError):
            logger.debug("Could not fetch model details for %s", info.name)
            return info

    async def get_version(self) -> str | None:
        # 版本只用于侧栏展示，失败不影响使用。
        import httpx

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
                response = await client.get(f"{self.base_url}/api/version")
                response.raise_for_status()
                version = response.json().get("version")
                return str(version) if version else None
        except (httpx.HTTPError, ValueError, TypeError):
            return None

    async def running_models(self) -> list[RunningModel]:
        """读取已加载模型；任何失败都返回空列表。"""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
                response = await client.get(f"{self.base_url}/api/ps")
                response.raise_for_status()
                payload = response.json()
                raw_models = payload.get("models") or []
        except (httpx.HTTPError, ValueError, TypeError):
            logger.debug("Could not read /api/ps", exc_info=True)
            return []

        running: list[RunningModel] = []
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "")
            if not name:
                continue
            details = item.get("details") or {}
            running.append(
                RunningModel(
                    name=name,
                    size=int(item.get("size") or 0),
                    size_vram=int(item.get("size_vram") or 0),
                    parameter_size=str(details.get("parameter_size") or ""),
                    quantization=str(details.get("quantization_level") or ""),
                    expires_at=str(item.get("expires_at") or ""),
                )
            )
        return running

    # ------------------------------------------------------------------
    # 聊天
    # ------------------------------------------------------------------
    async def chat_stream(
        self,
        model: str,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float = 0.7,
        think: bool | None = None,
        num_ctx: int | None = None,
        num_gpu: int | None = None,
    ) -> AsyncIterator[ChatChunk]:
        import httpx

        # 只传非空选项，未指定时让 Ollama 使用默认值。
        options: dict[str, Any] = {"temperature": float(temperature)}
        if num_ctx is not None:
            options["num_ctx"] = int(num_ctx)
        if num_gpu is not None:
            options["num_gpu"] = int(num_gpu)
        payload = {
            "model": model,
            "messages": list(messages),
            "stream": True,
            "options": options,
        }
        if think is not None:
            payload["think"] = bool(think)
        # 读超时留得较长，兼容大模型加载和长回答。
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=30.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as response:
                    if response.status_code >= 400:
                        raise OllamaError(
                            self._error_from_body(
                                await response.aread(),
                                response.status_code,
                                model,
                            )
                        )

                    # Ollama 用 NDJSON 流式返回，每行一个 JSON 对象。
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            logger.debug("Ignoring malformed stream line: %s", line[:200])
                            continue
                        if not isinstance(chunk, dict):
                            continue
                        if chunk.get("error"):
                            raise OllamaError(str(chunk["error"]))
                        message = chunk.get("message") or {}
                        content = str(message.get("content") or "")
                        thinking = str(message.get("thinking") or "")
                        done = bool(chunk.get("done"))
                        if content or thinking or done:
                            yield ChatChunk(
                                content=content,
                                thinking=thinking,
                                done=done,
                                eval_count=int(chunk.get("eval_count") or 0),
                                eval_duration_ns=int(
                                    chunk.get("eval_duration") or 0
                                ),
                                prompt_eval_count=int(
                                    chunk.get("prompt_eval_count") or 0
                                ),
                                prompt_eval_duration_ns=int(
                                    chunk.get("prompt_eval_duration") or 0
                                ),
                            )
                        if done:
                            return
        except httpx.ConnectError as exc:
            raise OllamaError(
                "无法连接到 Ollama。请确认服务正在运行"
                f"（{self.base_url}），可执行 `ollama serve` 启动。"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaError("等待 Ollama 响应超时，请稍后重试。") from exc
        except OllamaError:
            raise
        except httpx.HTTPError as exc:
            raise OllamaError(f"与 Ollama 的连接中断：{exc}") from exc

    @staticmethod
    def _error_from_body(body: bytes, status_code: int, model: str) -> str:
        """把 Ollama 的原始错误转换成中文提示。"""
        text = body.decode("utf-8", errors="replace")
        detail = ""
        error_obj: dict[str, Any] | None = None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                raw_error = parsed.get("error")
                if isinstance(raw_error, dict):
                    error_obj = raw_error
                    detail = str(raw_error.get("message") or "")
                else:
                    detail = str(raw_error or "")
        except json.JSONDecodeError:
            detail = text.strip()

        if error_obj:
            error_type = str(error_obj.get("type") or "")
            # 上下文超限给出具体 token 数和调整建议。
            if (
                "exceed_context_size" in error_type
                or "exceeds the available context size" in detail
            ):
                prompt_tokens = error_obj.get("n_prompt_tokens")
                context_tokens = error_obj.get("n_ctx")
                numbers = ""
                if prompt_tokens is not None and context_tokens is not None:
                    numbers = (
                        f"本次请求约 {prompt_tokens} tokens，而模型当前只分配了 "
                        f"{context_tokens} tokens 的上下文窗口。"
                    )
                else:
                    numbers = detail
                return (
                    f"上下文长度不足：{numbers}\n\n"
                    "请到「设置」把「上下文长度 (num_ctx)」调大"
                    "（例如 8192 或 16384），或新建一个对话后重试。"
                    "为了避免上下文膨胀，应用只会重复发送最近一次带图消息里的图片。"
                )

        if status_code == 404:
            return (
                f"模型 `{model}` 不存在或已被删除。"
                "请在终端执行 `ollama pull <模型名>` 后刷新模型列表。"
            )
        # 显存/内存不足时提示降低计算模式或上下文。
        lowered = detail.lower()
        if any(
            hint in lowered
            for hint in (
                "out of memory",
                "failed to allocate",
                "not enough memory",
                "exceeds available memory",
            )
        ):
            return (
                "显存或内存不足：当前计算模式无法把模型加载进去。\n\n"
                "请到「设置」把「计算模式」改为「自动」或「纯 CPU」，"
                "或降低「上下文长度」后重试。"
            )
        return f"Ollama 请求失败（HTTP {status_code}）：{detail or '未知错误'}"


def build_api_messages(
    messages: Sequence[Any],
    *,
    system_prompt: str = "",
    image_loader: Any | None = None,
) -> list[dict[str, Any]]:
    """把本地消息转换成 Ollama 的聊天请求格式。

    ``image_loader`` 负责读取附件字节。只为最近一条带图用户消息发送图片，
    避免历史截图反复占用上下文；旧图片仍保留在本地界面。
    """
    import base64

    payload: list[dict[str, Any]] = []
    if system_prompt.strip():
        payload.append({"role": "system", "content": system_prompt.strip()})

    # 找到最近一条带图用户消息的位置。
    latest_image_index = -1
    for index, message in enumerate(messages):
        if (
            getattr(message, "role", "") == "user"
            and getattr(message, "attachments", [])
        ):
            latest_image_index = index

    for index, message in enumerate(messages):
        role = getattr(message, "role", "")
        if role not in {"user", "assistant"}:
            continue
        # 失败回答可以在界面显示，但不能进入下一轮上下文。
        if getattr(message, "error", False):
            continue
        content = getattr(message, "content", "") or ""
        attachments = getattr(message, "attachments", []) if role == "user" else []
        if not content.strip() and not attachments:
            continue
        entry: dict[str, Any] = {"role": role, "content": content}
        if (
            attachments
            and image_loader is not None
            and index == latest_image_index
        ):
            images = []
            for attachment in attachments:
                try:
                    images.append(base64.b64encode(image_loader(attachment)).decode("ascii"))
                except OSError:
                    logger.warning("Skipping unreadable attachment %s", attachment.path)
            if images:
                entry["images"] = images
        payload.append(entry)
    return payload
