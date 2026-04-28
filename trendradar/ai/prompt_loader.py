# coding=utf-8
"""
提示词模板加载工具

从配置目录中加载 [system] / [user] 格式的提示词文件，
供 analyzer、translator、filter 等模块共享使用。
"""

from pathlib import Path
from typing import Tuple

# 项目 config 根目录
_CONFIG_ROOT = Path(__file__).parent.parent.parent / "config"


def _extract_section(
    content: str,
    section_name: str,
    next_sections: Tuple[str, ...],
) -> str:
    """Extract a named prompt section while keeping backward compatibility."""
    marker = f"[{section_name}]"
    start = content.find(marker)
    if start == -1:
        return ""

    section_start = start + len(marker)
    section_end = len(content)
    for next_section in next_sections:
        next_index = content.find(f"[{next_section}]", section_start)
        if next_index != -1:
            section_end = min(section_end, next_index)

    return content[section_start:section_end].strip()


def load_prompt_template(
    prompt_file: str,
    config_subdir: str = "",
    label: str = "AI",
) -> Tuple[str, str]:
    """
    加载提示词模板文件，解析 [system] 和 [user] 部分。

    Args:
        prompt_file: 提示词文件名
        config_subdir: config 下的子目录（如 "ai_filter"），为空则直接在 config/ 下查找
        label: 日志标签，用于提示文件缺失时的打印

    Returns:
        (system_prompt, user_prompt_template) 元组
    """
    config_dir = _CONFIG_ROOT / config_subdir if config_subdir else _CONFIG_ROOT
    prompt_path = config_dir / prompt_file

    if not prompt_path.exists():
        print(f"[{label}] 提示词文件不存在: {prompt_path}")
        return "", ""

    content = prompt_path.read_text(encoding="utf-8")

    system_prompt = ""
    user_prompt = ""

    if any(section in content for section in ("[persona]", "[system]", "[user]")):
        persona_prompt = _extract_section(content, "persona", ("system", "user"))
        system_body = _extract_section(content, "system", ("user",))
        user_prompt = _extract_section(content, "user", ("persona", "system", "user"))

        system_parts = []
        if persona_prompt:
            system_parts.append(persona_prompt)
        if system_body:
            system_parts.append(system_body)
        system_prompt = "\n\n".join(system_parts)
    else:
        user_prompt = content

    return system_prompt, user_prompt
