# coding=utf-8
"""
AI 客户端模块

基于 LiteLLM 的统一 AI 模型接口
支持 100+ AI 提供商（OpenAI、DeepSeek、Gemini、Claude、国内模型等）
"""

import os
from typing import Any, Dict, List

from litellm import completion

from trendradar.ai.mcp_bridge import run_mcp_completion


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
        self.api_key = config.get("API_KEY") or os.environ.get("AI_API_KEY", "")
        self.api_base = config.get("API_BASE", "")
        self.temperature = config.get("TEMPERATURE", 1.0)
        self.max_tokens = config.get("MAX_TOKENS", 5000)
        self.timeout = config.get("TIMEOUT", 120)
        self.num_retries = config.get("NUM_RETRIES", 2)
        self.fallback_models = config.get("FALLBACK_MODELS", [])
        self.extra_params = config.get("EXTRA_PARAMS", {}) or {}

    def _build_request_params(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Dict[str, Any]:
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

        merged_extra_params = {
            key: value
            for key, value in self.extra_params.items()
            if value is not None
        }
        for key, value in kwargs.items():
            if key not in reserved_keys and value is not None:
                merged_extra_params[key] = value

        params.update(merged_extra_params)
        return params

    def _extract_content(self, response: Any) -> str:
        content = response.choices[0].message.content
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        return content or ""

    def _chat_with_mcp(
        self,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
        mcp_config: Dict[str, Any],
    ) -> str:
        completion_params = dict(params)
        completion_params.pop("messages", None)
        return run_mcp_completion(messages, completion_params, mcp_config)

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
        params = self._build_request_params(messages, **kwargs)

        if isinstance(mcp_config, dict) and mcp_config.get("ENABLED"):
            try:
                return self._chat_with_mcp(messages, params, mcp_config)
            except Exception as exc:
                print(f"[AI] MCP bridge unavailable, fallback to plain completion: {type(exc).__name__}: {exc}")

        response = completion(**params)
        return self._extract_content(response)

    def validate_config(self) -> tuple[bool, str]:
        """
        验证配置是否有效

        Returns:
            tuple: (是否有效, 错误信息)
        """
        if not self.model:
            return False, "未配置 AI 模型（model）"

        if not self.api_key:
            return False, "未配置 AI API Key，请在 config.yaml 或环境变量 AI_API_KEY 中设置"

        # 验证模型格式（应该包含 provider/model）
        if "/" not in self.model:
            return False, f"模型格式错误: {self.model}，应为 'provider/model' 格式（如 'deepseek/deepseek-chat'）"

        return True, ""
