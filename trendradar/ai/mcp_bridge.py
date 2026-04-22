# coding=utf-8

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_vscode_mcp_server_config(config_file: str, server_name: str) -> Optional[Dict[str, Any]]:
    path = Path(config_file)
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    servers = payload.get("servers") or payload.get("mcpServers") or {}
    if not isinstance(servers, dict):
        return None

    server = servers.get(server_name)
    if not isinstance(server, dict):
        return None
    if server.get("type", "stdio") != "stdio":
        return None

    command = str(server.get("command", "")).strip()
    if not command:
        return None

    args = server.get("args", [])
    env = server.get("env", {})
    if not isinstance(args, list):
        args = []
    if not isinstance(env, dict):
        env = {}

    return {
        "command": command,
        "args": [str(arg) for arg in args],
        "env": {str(key): str(value) for key, value in env.items() if value is not None},
    }


def build_runtime_mcp_server_config(
    runtime_config: Dict[str, Any],
    server_name: str,
) -> Optional[Dict[str, Any]]:
    normalized_server = str(server_name or "").strip().lower()
    if normalized_server != "minimax":
        return None

    api_key = str(os.environ.get("AI_API_KEY") or runtime_config.get("API_KEY") or "").strip()
    api_host = str(runtime_config.get("API_HOST") or runtime_config.get("MCP_API_HOST") or "").strip()
    if not api_key or not api_host:
        return None

    return {
        "command": "uvx",
        "args": ["minimax-coding-plan-mcp", "-y"],
        "env": {
            "MINIMAX_API_KEY": api_key,
            "MINIMAX_API_HOST": api_host,
        },
    }


def resolve_mcp_server_config(
    runtime_config: Dict[str, Any],
    server_name: str,
) -> Optional[Dict[str, Any]]:
    runtime_server_config = build_runtime_mcp_server_config(runtime_config, server_name)
    if runtime_server_config:
        return runtime_server_config

    config_file = str(runtime_config.get("CONFIG_FILE") or "").strip()
    if not config_file:
        return None

    return load_vscode_mcp_server_config(config_file, server_name)


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
                continue
            text = getattr(item, "text", "")
            if text:
                parts.append(str(text))
                continue
            parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _dump_tool_call(tool_call: Any) -> Dict[str, Any]:
    if hasattr(tool_call, "model_dump"):
        return tool_call.model_dump(exclude_none=True)
    if isinstance(tool_call, dict):
        return tool_call

    function = getattr(tool_call, "function", None)
    return {
        "id": getattr(tool_call, "id", ""),
        "type": getattr(tool_call, "type", "function"),
        "function": {
            "name": getattr(function, "name", ""),
            "arguments": getattr(function, "arguments", "{}"),
        },
    }


def _extract_tool_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        function = tool_call.get("function") or {}
        return str(function.get("name", ""))
    function = getattr(tool_call, "function", None)
    return str(getattr(function, "name", "") or "")


def _stringify_tool_result(tool_result: Any) -> str:
    content = getattr(tool_result, "content", None)
    if isinstance(content, list):
        parts = []
        for item in content:
            text = getattr(item, "text", "")
            if text:
                parts.append(str(text))
                continue
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item.get("text")))
                continue
            if hasattr(item, "model_dump"):
                parts.append(json.dumps(item.model_dump(exclude_none=True), ensure_ascii=False))
                continue
            parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    if hasattr(tool_result, "model_dump"):
        dumped = tool_result.model_dump(exclude_none=True)
        return json.dumps(dumped, ensure_ascii=False)

    return str(tool_result or "")


async def _run_mcp_completion_async(
    messages: List[Dict[str, Any]],
    completion_params: Dict[str, Any],
    runtime_config: Dict[str, Any],
) -> str:
    from litellm import acompletion
    from litellm.experimental_mcp_client import call_openai_tool, load_mcp_tools
    from litellm.experimental_mcp_client.client import MCPClient
    from litellm.types.mcp import MCPTransport

    server_names = runtime_config.get("SERVERS") or ["MiniMax"]
    max_tool_rounds = max(1, int(runtime_config.get("MAX_TOOL_ROUNDS", 4)))
    last_error: Optional[Exception] = None

    for server_name in server_names:
        server_config = resolve_mcp_server_config(runtime_config, str(server_name))
        if not server_config:
            continue

        client = MCPClient(
            transport_type=MCPTransport.stdio,
            timeout=float(completion_params.get("timeout", 120)),
            stdio_config=server_config,
        )

        async def _run_with_session(session):
            tools = await load_mcp_tools(session, format="openai")
            if not tools:
                raise RuntimeError(f"MCP server {server_name} has no tools")

            conversation = [dict(message) for message in messages]
            last_text = ""
            request_params = dict(completion_params)

            for _ in range(max_tool_rounds + 1):
                response = await acompletion(
                    **request_params,
                    messages=conversation,
                    tools=tools,
                    tool_choice="auto",
                )

                message = response.choices[0].message
                last_text = _coerce_message_content(getattr(message, "content", ""))
                tool_calls = getattr(message, "tool_calls", None) or []
                if not tool_calls:
                    return last_text

                conversation.append(
                    {
                        "role": "assistant",
                        "content": last_text,
                        "tool_calls": [_dump_tool_call(tool_call) for tool_call in tool_calls],
                    }
                )

                for tool_call in tool_calls:
                    tool_result = await call_openai_tool(session, tool_call)
                    conversation.append(
                        {
                            "role": "tool",
                            "tool_call_id": getattr(tool_call, "id", ""),
                            "name": _extract_tool_name(tool_call),
                            "content": _stringify_tool_result(tool_result),
                        }
                    )

            return last_text

        try:
            return await client.run_with_session(_run_with_session)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("No available MCP server configuration found")


def run_mcp_completion(
    messages: List[Dict[str, Any]],
    completion_params: Dict[str, Any],
    runtime_config: Dict[str, Any],
) -> str:
    return asyncio.run(
        _run_mcp_completion_async(messages, completion_params, runtime_config)
    )