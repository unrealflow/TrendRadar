# coding=utf-8
"""
AI 客户端模块

基于 LiteLLM 的统一 AI 模型接口
支持 100+ AI 提供商（OpenAI、DeepSeek、Gemini、Claude、国内模型等）
"""

import os
import threading
import time
from typing import Any, Dict, List, Optional

from litellm import completion
from openai import OpenAI

from trendradar.ai.mcp_bridge import run_mcp_completion


# BEGIN BY wangsikan@kuaishou.com: heartbeat logging for long AI calls
def _start_heartbeat(
    label: str,
    interval: float = 15.0,
    last_activity: Optional[List[float]] = None,
) -> threading.Event:
    """Start a background heartbeat that prints elapsed time every `interval` seconds.

    Returns an Event; set it to stop the heartbeat thread.
    """
    stop_event = threading.Event()
    start_ts = time.monotonic()

    def _tick() -> None:
        while not stop_event.wait(interval):
            elapsed = time.monotonic() - start_ts
            if last_activity is not None:
                idle_for = time.monotonic() - last_activity[0]
                if idle_for < interval:
                    continue
                print(
                    f"[AI][heartbeat] {label} waiting for next chunk... "
                    f"elapsed={elapsed:.1f}s idle={idle_for:.1f}s",
                    flush=True,
                )
                continue
            print(f"[AI][heartbeat] {label} still waiting... elapsed={elapsed:.1f}s", flush=True)

    thread = threading.Thread(target=_tick, name=f"ai-heartbeat-{label}", daemon=True)
    thread.start()
    return stop_event
# END BY wangsikan@kuaishou.com


class AIClient:
    """统一的 AI 客户端（基于 LiteLLM）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 AI 客户端

        Args:
            config: AI 配置字典
                - MODEL: 模型标识（格式: provider/model_name）
                - API_KEY: API 密钥
                - API_BASE: API 基础 URL（可选）
                - TEMPERATURE: 采样温度
                - MAX_TOKENS: 最大生成 token 数
                - TIMEOUT: 请求超时时间（秒）
                - NUM_RETRIES: 重试次数（可选）
                - FALLBACK_MODELS: 备用模型列表（可选）
        """
        self.model = config.get("MODEL", "deepseek/deepseek-chat")
        configured_api_key = config.get("API_KEY")
        if str(self.model).lower().startswith("github_copilot/"):
            self.api_key = configured_api_key if configured_api_key is not None else ""
        else:
            self.api_key = configured_api_key or os.environ.get("AI_API_KEY", "")
        self.api_base = config.get("API_BASE", "")
        self.temperature = config.get("TEMPERATURE", 1.0)
        self.max_tokens = config.get("MAX_TOKENS", 5000)
        self.timeout = config.get("TIMEOUT", 120)
        self.num_retries = config.get("NUM_RETRIES", 2)
        self.fallback_models = config.get("FALLBACK_MODELS", [])
        self.extra_params = config.get("EXTRA_PARAMS", {}) or {}
        self.last_reasoning_content = ""
        self.last_output_content = ""
        self.last_display_response = ""
        self.last_finish_reason = ""

    def _should_request_reasoning(self, model: str, params: Dict[str, Any]) -> bool:
        normalized_model = str(model or "").lower()
        if not normalized_model:
            return False
        if "reasoning_effort" in params or "thinking" in params:
            return False
        return "gpt-5" in normalized_model

    def _should_use_minimax_openai_sdk(self, params: Dict[str, Any]) -> bool:
        model = str(params.get("model", self.model) or "").strip().lower()
        api_base = str(params.get("api_base", self.api_base) or "").strip().lower()
        return model.startswith("openai/minimax-") and "api.minimaxi.com/v1" in api_base

    def _should_stream_reasoning(self, params: Dict[str, Any]) -> bool:
        model = str(params.get("model", self.model) or "").strip().lower()
        if not model:
            return False
        return (
            "gpt-5" in model
            or model.startswith("openai/minimax-")
            or "reasoning_effort" in params
            or "thinking" in params
        )

    def _get_value(self, value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    def _normalize_openai_model_name(self, model: str) -> str:
        normalized_model = str(model or "")
        if normalized_model.lower().startswith("openai/"):
            return normalized_model.split("/", 1)[1]
        return normalized_model

    def _chat_with_minimax_openai_sdk(self, params: Dict[str, Any], stream: bool = False) -> Any:
        client = OpenAI(
            api_key=params.get("api_key") or self.api_key,
            base_url=params.get("api_base") or self.api_base or "https://api.minimaxi.com/v1",
        )
        request_params: Dict[str, Any] = {
            "model": self._normalize_openai_model_name(str(params.get("model", self.model))),
            "messages": params.get("messages", []),
        }

        for key in ("temperature", "max_tokens", "top_p", "timeout"):
            value = params.get(key)
            if value is not None:
                request_params[key] = value

        extra_body = dict(params.get("extra_body") or {})
        extra_body.setdefault("reasoning_split", True)
        request_params["extra_body"] = extra_body
        if stream:
            request_params["stream"] = True

        return client.chat.completions.create(**request_params)

    def _normalize_response_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                text = ""
                if isinstance(item, dict):
                    item_type = str(item.get("type") or "").strip().lower()
                    if item_type == "thinking":
                        continue
                    text = item.get("text") or item.get("content") or ""
                else:
                    text = str(item)
                text = str(text).strip()
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return str(value)

    def _extract_reasoning_from_payload(self, payload: Any) -> str:
        reasoning_content = self._normalize_response_text(
            self._get_value(payload, "reasoning_content", None)
        ).strip()
        if reasoning_content:
            return reasoning_content

        thinking_blocks = self._get_value(payload, "thinking_blocks", None)
        parts = []
        if isinstance(thinking_blocks, list):
            for block in thinking_blocks:
                text = ""
                if isinstance(block, dict):
                    text = block.get("thinking") or block.get("text") or ""
                else:
                    text = getattr(block, "thinking", None) or getattr(block, "text", None) or ""
                text = str(text).strip()
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)

        reasoning_details = self._get_value(payload, "reasoning_details", None)
        if isinstance(reasoning_details, list):
            for detail in reasoning_details:
                text = ""
                if isinstance(detail, dict):
                    text = detail.get("text") or ""
                else:
                    text = getattr(detail, "text", None) or ""
                text = str(text).strip()
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)

        message_content = self._get_value(payload, "content", None)
        if not isinstance(message_content, list):
            return ""

        for block in message_content:
            if not isinstance(block, dict):
                continue
            if str(block.get("type") or "").strip().lower() != "thinking":
                continue
            text = str(block.get("thinking") or block.get("text") or "").strip()
            if text:
                parts.append(text)

        return "\n".join(parts)

    def _extract_reasoning_content(self, response: Any) -> str:
        try:
            message = response.choices[0].message  # type: ignore[attr-defined]
        except Exception:
            return ""
        return self._extract_reasoning_from_payload(message)

    def _extract_stream_chunk(self, chunk: Any) -> tuple[str, str, str, Any]:
        choices = self._get_value(chunk, "choices", None)
        if not choices:
            return "", "", "", self._get_value(chunk, "usage", None)

        first_choice = choices[0]
        delta = self._get_value(first_choice, "delta", None) or self._get_value(first_choice, "message", None)
        if delta is None:
            return "", "", str(self._get_value(first_choice, "finish_reason", "") or ""), self._get_value(chunk, "usage", None)

        reasoning_piece = self._extract_reasoning_from_payload(delta)
        output_piece = self._normalize_response_text(self._get_value(delta, "content", None))
        finish_reason = str(self._get_value(first_choice, "finish_reason", "") or "")
        return reasoning_piece, output_piece, finish_reason, self._get_value(chunk, "usage", None)

    def _build_usage_str(self, usage: Any) -> str:
        if usage is None:
            return ""

        prompt_tokens = self._get_value(usage, "prompt_tokens", "?")
        completion_tokens = self._get_value(usage, "completion_tokens", "?")
        total_tokens = self._get_value(usage, "total_tokens", "?")
        if prompt_tokens == completion_tokens == total_tokens == "?":
            return ""
        return (
            f" prompt_tokens={prompt_tokens}"
            f" completion_tokens={completion_tokens}"
            f" total_tokens={total_tokens}"
        )

    def _stream_completion(self, params: Dict[str, Any], label: str) -> tuple[str, str, Any, str, bool]:
        stream_params = dict(params)
        stream_params["stream"] = True
        last_activity = [time.monotonic()]
        stop_event = _start_heartbeat(label, last_activity=last_activity)
        reasoning_parts: List[str] = []
        output_parts: List[str] = []
        finish_reason = ""
        usage = None
        trace_emitted = False
        reasoning_header_printed = False
        output_header_printed = False

        try:
            if self._should_use_minimax_openai_sdk(stream_params):
                stream = self._chat_with_minimax_openai_sdk(stream_params, stream=True)
            else:
                stream = completion(**stream_params)

            # Some SDKs may ignore stream=True and return a buffered response object directly.
            if self._get_value(stream, "choices", None):
                usage = self._get_value(stream, "usage", None)
                choices = self._get_value(stream, "choices", []) or []
                if choices:
                    finish_reason = str(self._get_value(choices[0], "finish_reason", "") or "")
                return (
                    self._extract_reasoning_content(stream),
                    self._extract_content(stream),
                    usage,
                    finish_reason,
                    False,
                )

            for chunk in stream:
                reasoning_piece, output_piece, chunk_finish_reason, chunk_usage = self._extract_stream_chunk(chunk)
                if chunk_finish_reason:
                    finish_reason = chunk_finish_reason
                if chunk_usage is not None:
                    usage = chunk_usage

                if reasoning_piece:
                    if not reasoning_header_printed:
                        print("[AI][trace] === Thinking / Reasoning Content ===", flush=True)
                        reasoning_header_printed = True
                        trace_emitted = True
                    print(reasoning_piece, end="", flush=True)
                    reasoning_parts.append(reasoning_piece)
                    last_activity[0] = time.monotonic()

                if output_piece:
                    if not output_header_printed:
                        if reasoning_header_printed:
                            print(flush=True)
                        print("[AI][trace] === Output ===", flush=True)
                        output_header_printed = True
                        trace_emitted = True
                    print(output_piece, end="", flush=True)
                    output_parts.append(output_piece)
                    last_activity[0] = time.monotonic()
        finally:
            stop_event.set()

        if trace_emitted:
            print(flush=True)

        return "".join(reasoning_parts), "".join(output_parts), usage, finish_reason, trace_emitted

    def _cache_trace(
        self,
        reasoning_content: str,
        output_content: str,
        unavailable_reason: Optional[str] = None,
    ) -> tuple[str, str]:
        reasoning_text = (reasoning_content or "").strip()
        if not reasoning_text:
            reasoning_text = unavailable_reason or "(Not available from current model/provider)"

        output_text = (output_content or "").strip()
        if not output_text:
            output_text = "(Empty output)"

        self.last_reasoning_content = (reasoning_content or "").strip()
        self.last_output_content = output_text
        self.last_display_response = (
            "Thinking / Reasoning Content\n"
            f"{reasoning_text}\n\n"
            "Output\n"
            f"{output_text}"
        )
        return reasoning_text, output_text

    def _emit_trace(self, reasoning_text: str, output_text: str) -> None:
        print("[AI][trace] === Thinking / Reasoning Content ===", flush=True)
        print(reasoning_text, flush=True)
        print("[AI][trace] === Output ===", flush=True)
        print(output_text, flush=True)

    def _build_request_params(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Dict[str, Any]:
        skip_reasoning = bool(kwargs.pop("_skip_reasoning", False))
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "timeout": kwargs.get("timeout", self.timeout),
            "num_retries": kwargs.get("num_retries", self.num_retries),
        }

        if self.api_key:
            params["api_key"] = self.api_key

        if self.api_base:
            params["api_base"] = self.api_base

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens and max_tokens > 0:
            params["max_tokens"] = max_tokens

        if self.fallback_models:
            params["fallbacks"] = self.fallback_models

        reserved_keys = {
            "model",
            "messages",
            "temperature",
            "timeout",
            "num_retries",
            "api_key",
            "api_base",
            "max_tokens",
            "fallbacks",
            "mcp_config",
        }

        _reasoning_keys = {"reasoning_effort", "thinking"}
        merged_extra_params = {
            key: value
            for key, value in self.extra_params.items()
            if value is not None and (not skip_reasoning or key not in _reasoning_keys)
        }
        for key, value in kwargs.items():
            if key not in reserved_keys and value is not None:
                if skip_reasoning and key in _reasoning_keys:
                    continue
                merged_extra_params[key] = value

        params.update(merged_extra_params)
        if not skip_reasoning and self._should_request_reasoning(str(params.get("model", self.model)), params):
            params["reasoning_effort"] = {"effort": "low", "summary": "detailed"}
            params["drop_params"] = True
        elif not skip_reasoning and ("reasoning_effort" in params or "thinking" in params):
            params.setdefault("drop_params", True)
        return params

    def _extract_content(self, response: Any) -> str:
        # BEGIN BY wangsikan@kuaishou.com: defensive handling for empty/invalid response
        choices = getattr(response, "choices", None)
        if not choices:
            # LiteLLM may return a response without choices when auth fails (e.g. github_copilot
            # OAuth device flow not completed) or the upstream provider returns an error body.
            finish_hint = ""
            try:
                finish_hint = f" raw={str(response)[:200]}"
            except Exception:
                pass
            raise RuntimeError(
                "AI 响应为空（choices 列表为空）。常见原因：上游鉴权失败/超时、"
                "或 github_copilot 设备流未完成登录。请检查日志中上方的鉴权提示。"
                + finish_hint
            )
        first = choices[0]
        message = getattr(first, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        # END BY wangsikan@kuaishou.com
        return self._normalize_response_text(content)

    def _chat_with_mcp(
        self,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
        mcp_config: Dict[str, Any],
    ) -> str:
        completion_params = dict(params)
        completion_params.pop("messages", None)
        # BEGIN CHANGE BY wangsikan@kuaishou.com: capture reasoning returned by MCP bridge
        reasoning, output = run_mcp_completion(messages, completion_params, mcp_config)
        self.last_reasoning_content = reasoning
        return output
        # END CHANGE BY wangsikan@kuaishou.com

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        调用 AI 模型进行对话

        Args:
            messages: 消息列表，格式: [{"role": "system/user/assistant", "content": "..."}]
            **kwargs: 额外参数，会覆盖默认配置

        Returns:
            str: AI 响应内容

        Raises:
            Exception: API 调用失败时抛出异常
        """
        mcp_config = kwargs.pop("mcp_config", None)
        log_label = kwargs.pop("_log_label", None)
        skip_reasoning = bool(kwargs.pop("_skip_reasoning", False))
        self.last_reasoning_content = ""
        self.last_output_content = ""
        self.last_display_response = ""
        self.last_finish_reason = ""
        params = self._build_request_params(messages, _skip_reasoning=skip_reasoning, **kwargs)

        if isinstance(mcp_config, dict) and mcp_config.get("ENABLED"):
            try:
                output = self._chat_with_mcp(messages, params, mcp_config)
                # BEGIN CHANGE BY wangsikan@kuaishou.com: surface reasoning from MCP bridge
                mcp_reasoning = self.last_reasoning_content or ""
                reasoning_text, output_text = self._cache_trace(
                    mcp_reasoning,
                    output,
                    unavailable_reason=None if mcp_reasoning else "(Reasoning not returned by MCP provider)",
                )
                # END CHANGE BY wangsikan@kuaishou.com
                self.last_finish_reason = ""
                self._emit_trace(reasoning_text, output_text)
                return output
            except Exception as exc:
                print(f"[AI] MCP bridge unavailable, fallback to plain completion: {type(exc).__name__}: {exc}")

        # BEGIN BY wangsikan@kuaishou.com: detailed logging around completion call
        msg_count = len(messages)
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        label = log_label or "completion"
        print(
            f"[AI][call] {label} start model={params.get('model')} "
            f"messages={msg_count} prompt_chars={prompt_chars} "
            f"timeout={params.get('timeout')} max_tokens={params.get('max_tokens')}",
            flush=True,
        )
        start_ts = time.monotonic()
        response = None
        streamed_trace_emitted = False
        content = ""
        reasoning_content = ""
        usage = None
        finish_reason = ""
        try:
            if self._should_stream_reasoning(params):
                try:
                    reasoning_content, content, usage, finish_reason, streamed_trace_emitted = self._stream_completion(params, label)
                except Exception as stream_exc:
                    print(
                        f"[AI][call] {label} stream unavailable, fallback to buffered completion: "
                        f"{type(stream_exc).__name__}: {str(stream_exc)[:200]}",
                        flush=True,
                    )
                    stop_event = _start_heartbeat(label)
                    try:
                        if self._should_use_minimax_openai_sdk(params):
                            response = self._chat_with_minimax_openai_sdk(params)
                        else:
                            response = completion(**params)
                    finally:
                        stop_event.set()
            else:
                stop_event = _start_heartbeat(label)
                try:
                    if self._should_use_minimax_openai_sdk(params):
                        response = self._chat_with_minimax_openai_sdk(params)
                    else:
                        response = completion(**params)
                finally:
                    stop_event.set()
        except Exception as exc:
            elapsed = time.monotonic() - start_ts
            print(
                f"[AI][call] {label} FAILED after {elapsed:.1f}s: "
                f"{type(exc).__name__}: {str(exc)[:200]}",
                flush=True,
            )
            raise

        elapsed = time.monotonic() - start_ts
        if response is not None:
            usage = getattr(response, "usage", None)
            try:
                finish_reason = response.choices[0].finish_reason  # type: ignore[attr-defined]
            except Exception:
                finish_reason = ""
            content = self._extract_content(response)
            reasoning_content = self._extract_reasoning_content(response)

        usage_str = self._build_usage_str(usage)
        self.last_finish_reason = str(finish_reason or "")
        reasoning_text, output_text = self._cache_trace(reasoning_content, content)
        print(
            f"[AI][call] {label} done elapsed={elapsed:.1f}s "
            f"finish_reason={self.last_finish_reason or None} content_chars={len(content)}{usage_str}",
            flush=True,
        )
        if not streamed_trace_emitted:
            self._emit_trace(reasoning_text, output_text)
        return content
        # END BY wangsikan@kuaishou.com

    def validate_config(self) -> tuple[bool, str]:
        """
        验证配置是否有效

        Returns:
            tuple: (是否有效, 错误信息)
        """
        if not self.model:
            return False, "未配置 AI 模型（model）"

        # BEGIN BY wangsikan@kuaishou.com: github_copilot uses OAuth device flow, no API key required
        if not self.api_key and not str(self.model).lower().startswith("github_copilot/"):
            return False, "未配置 AI API Key，请在 config.yaml 或环境变量 AI_API_KEY 中设置"
        # END BY wangsikan@kuaishou.com

        # 验证模型格式（应该包含 provider/model）
        if "/" not in self.model:
            return False, f"模型格式错误: {self.model}，应为 'provider/model' 格式（如 'deepseek/deepseek-chat'）"

        return True, ""
