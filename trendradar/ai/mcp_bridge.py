# coding=utf-8

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# BEGIN BY wangsikan@kuaishou.com: Phase 3 - observable MCP bridge
# Shared heartbeat so the outer watchdog in analyzer can tell whether MCP is
# still making progress (tool calls, LLM round trips, etc.).
MCP_LAST_HEARTBEAT = {"ts": time.time(), "stage": "idle", "detail": ""}
_MCP_HEARTBEAT_LOCK = threading.Lock()


def _heartbeat(stage: str, detail: str = "") -> None:
    with _MCP_HEARTBEAT_LOCK:
        MCP_LAST_HEARTBEAT["ts"] = time.time()
        MCP_LAST_HEARTBEAT["stage"] = stage
        MCP_LAST_HEARTBEAT["detail"] = detail


def mcp_seconds_since_heartbeat() -> float:
    with _MCP_HEARTBEAT_LOCK:
        return time.time() - MCP_LAST_HEARTBEAT["ts"]


def mcp_current_stage() -> Dict[str, Any]:
    with _MCP_HEARTBEAT_LOCK:
        return dict(MCP_LAST_HEARTBEAT)


def _log(message: str) -> None:
    """Flush-line logging so Windows PowerShell tailing shows live progress."""
    print(f"[MCP] {message}", flush=True)
    try:
        sys.stdout.flush()
    except Exception:
        pass
# END BY wangsikan@kuaishou.com


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


def _extract_reasoning_from_message(message: Any) -> str:
    """Extract reasoning/thinking content from an LLM response message."""
    return str(getattr(message, "reasoning_content", "") or "").strip()


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
) -> tuple:
    from litellm import acompletion
    from litellm.experimental_mcp_client import call_openai_tool, load_mcp_tools
    from litellm.experimental_mcp_client.client import MCPClient
    from litellm.types.mcp import MCPTransport

    server_names = runtime_config.get("SERVERS") or ["MiniMax"]
    max_tool_rounds = max(1, int(runtime_config.get("MAX_TOOL_ROUNDS", 4)))
    last_error: Optional[Exception] = None
    # BEGIN BY wangsikan@kuaishou.com: Phase 3 - detailed MCP logging
    _heartbeat("resolve_servers", f"candidates={list(server_names)}")
    _log(f"候选 server: {list(server_names)} | max_tool_rounds={max_tool_rounds}")
    # END BY wangsikan@kuaishou.com

    for server_name in server_names:
        server_config = resolve_mcp_server_config(runtime_config, str(server_name))
        if not server_config:
            _log(f"server={server_name} 未解析到配置，跳过")
            continue

        # BEGIN BY wangsikan@kuaishou.com: Phase 3 - log command being launched
        _log(
            f"启动 server={server_name} command={server_config.get('command')} "
            f"args={server_config.get('args')}"
        )
        _heartbeat("spawn", f"server={server_name}")
        spawn_start = time.time()
        # END BY wangsikan@kuaishou.com

        client = MCPClient(
            transport_type=MCPTransport.stdio,
            timeout=float(completion_params.get("timeout", 120)),
            stdio_config=server_config,
        )

        async def _run_with_session(session):
            # BEGIN BY wangsikan@kuaishou.com: Phase 3 - log tool loading + per-round timings
            load_start = time.time()
            _heartbeat("load_tools", f"server={server_name}")
            tools = await load_mcp_tools(session, format="openai")
            _log(
                f"server={server_name} 工具加载完成 "
                f"(耗时 {time.time() - load_start:.1f}s, tools={len(tools or [])}, "
                f"冷启动总耗时 {time.time() - spawn_start:.1f}s)"
            )
            # END BY wangsikan@kuaishou.com
            if not tools:
                raise RuntimeError(f"MCP server {server_name} has no tools")

            conversation = [dict(message) for message in messages]
            last_text = ""
            last_reasoning = ""
            request_params = dict(completion_params)

            for round_idx in range(max_tool_rounds + 1):
                # BEGIN BY wangsikan@kuaishou.com: Phase 3 - per-round log + heartbeat
                round_start = time.time()
                _heartbeat("llm_call", f"round={round_idx}")
                _log(f"第 {round_idx} 轮 acompletion 发起...")
                # END BY wangsikan@kuaishou.com
                response = await acompletion(
                    **request_params,
                    messages=conversation,
                    tools=tools,
                    tool_choice="auto",
                )

                message = response.choices[0].message
                last_text = _coerce_message_content(getattr(message, "content", ""))
                last_reasoning = _extract_reasoning_from_message(message)
                tool_calls = getattr(message, "tool_calls", None) or []
                # BEGIN BY wangsikan@kuaishou.com: Phase 3 - round done log
                _log(
                    f"第 {round_idx} 轮 LLM 返回 (耗时 {time.time() - round_start:.1f}s, "
                    f"tool_calls={len(tool_calls)}, reasoning_chars={len(last_reasoning)})"
                )
                _heartbeat("llm_done", f"round={round_idx} tools={len(tool_calls)}")
                # END BY wangsikan@kuaishou.com
                if not tool_calls:
                    return last_reasoning, last_text

                conversation.append(
                    {
                        "role": "assistant",
                        "content": last_text,
                        "tool_calls": [_dump_tool_call(tool_call) for tool_call in tool_calls],
                    }
                )

                for tool_call in tool_calls:
                    # BEGIN BY wangsikan@kuaishou.com: Phase 3 - per-tool log
                    tool_name = _extract_tool_name(tool_call) or "<unknown>"
                    tool_start = time.time()
                    _heartbeat("tool_call", f"round={round_idx} tool={tool_name}")
                    _log(f"  调用工具 {tool_name} ...")
                    # END BY wangsikan@kuaishou.com
                    tool_result = await call_openai_tool(session, tool_call)
                    # BEGIN BY wangsikan@kuaishou.com: Phase 3 - tool result log
                    _log(
                        f"  工具 {tool_name} 完成 (耗时 {time.time() - tool_start:.1f}s)"
                    )
                    _heartbeat(
                        "tool_done",
                        f"round={round_idx} tool={tool_name}",
                    )
                    # END BY wangsikan@kuaishou.com
                    conversation.append(
                        {
                            "role": "tool",
                            "tool_call_id": getattr(tool_call, "id", ""),
                            "name": _extract_tool_name(tool_call),
                            "content": _stringify_tool_result(tool_result),
                        }
                    )

            return last_reasoning, last_text

        try:
            return await client.run_with_session(_run_with_session)
        except Exception as exc:
            # BEGIN BY wangsikan@kuaishou.com: Phase 3 - surface failure per server
            _log(f"server={server_name} 调用失败: {type(exc).__name__}: {exc}")
            # END BY wangsikan@kuaishou.com
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("No available MCP server configuration found")


def run_mcp_completion(
    messages: List[Dict[str, Any]],
    completion_params: Dict[str, Any],
    runtime_config: Dict[str, Any],
) -> tuple:
    # BEGIN BY wangsikan@kuaishou.com: Phase 3 - idle watchdog around asyncio.run
    idle_limit = float(runtime_config.get("IDLE_TIMEOUT_SECONDS", 300))

    async def _driver() -> tuple:
        main_task = asyncio.create_task(
            _run_mcp_completion_async(messages, completion_params, runtime_config)
        )
        _heartbeat("driver_start", f"idle_limit={idle_limit:.0f}s")
        while True:
            done, _pending = await asyncio.wait({main_task}, timeout=30)
            if main_task in done:
                return main_task.result()
            idle = mcp_seconds_since_heartbeat()
            stage = mcp_current_stage()
            _log(
                f"watchdog 心跳检查 idle={idle:.0f}s stage={stage.get('stage')} "
                f"detail={stage.get('detail')}"
            )
            if idle > idle_limit:
                _log(
                    f"超过 {idle_limit:.0f}s 无心跳变化，主动取消当前 MCP 调用"
                )
                main_task.cancel()
                try:
                    await main_task
                except asyncio.CancelledError:
                    pass
                raise TimeoutError(
                    f"MCP 调用 {idle:.0f}s 无心跳，触发看门狗中断 "
                    f"(stage={stage.get('stage')} detail={stage.get('detail')})"
                )

    return asyncio.run(_driver())
    # END BY wangsikan@kuaishou.com