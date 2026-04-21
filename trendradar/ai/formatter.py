# coding=utf-8
"""
AI 分析结果格式化模块

将 AI 分析结果格式化为各推送渠道的样式
"""

import html as html_lib
import re
from .analyzer import AIAnalysisResult


_FEISHU_MARKDOWN_REPLACEMENTS = {
    "**核心判断**": "**🧠 核心判断**",
    "**重点消息**": "**🔥 重点消息**",
    "**评级建议**": "**📊 评级建议**",
    "**消息行业映射**": "**🔥 消息行业映射**",
    "**个人层**": "**🙋 个人层**",
    "**地区层**": "**🌍 地区层**",
    "**行业层**": "**🏭 行业层**",
    "**国家层**": "**🏛️ 国家层**",
    "**科技层**": "**🧪 科技层**",
    "**持仓汇总**": "**📦 持仓汇总**",
    "【持仓内】": "【🧺 持仓内】",
    "【持仓外】": "【🧭 持仓外】",
    "【风险提示】": "【⚠️ 风险提示】",
}


def _escape_html(text: str) -> str:
    """转义 HTML 特殊字符，防止 XSS 攻击"""
    return html_lib.escape(text) if text else ""


def _format_list_content(text: str) -> str:
    """
    格式化列表内容，确保序号前有换行
    例如将 "1. xxx 2. yyy" 转换为:
    1. xxx
    2. yyy
    """
    if not text:
        return ""
    
    # 去除首尾空白，防止 AI 返回的内容开头就有换行导致显示空行
    text = text.strip()

    # 0. 合并序号与紧随的【标签】（防御性处理）
    # 将 "1.\n【投资者】：" 或 "1. 【投资者】：" 合并为 "1. 投资者："
    text = re.sub(r'(\d+\.)\s*【([^】]+)】([:：]?)', r'\1 \2：', text)

    # 1. 规范化：确保 "1." 后面有空格
    result = re.sub(r'(\d+)\.([^ \d])', r'\1. \2', text)

    # 2. 强制换行：匹配 "数字."，且前面不是换行符
    #    (?!\d) 排除版本号/小数（如 2.0、3.5），避免将其误判为列表序号
    result = re.sub(r'(?<=[^\n])\s+(\d+\.)(?!\d)', r'\n\1', result)
    
    # 3. 处理 "1.**粗体**" 这种情况（虽然 Prompt 要求不输出 Markdown，但防御性处理）
    result = re.sub(r'(?<=[^\n])(\d+\.\*\*)', r'\n\1', result)

    # 4. 处理中文标点后的换行（排除版本号/小数）
    result = re.sub(r'([：:;,。；，])\s*(\d+\.)(?!\d)', r'\1\n\2', result)

    # 5. 处理 "XX方面："、"XX领域：" 等子标题换行
    # 只有在中文标点（句号、逗号、分号等）后才触发换行，避免破坏 "1. XX领域：" 格式
    result = re.sub(r'([。！？；，、])\s*([a-zA-Z0-9\u4e00-\u9fa5]+(方面|领域)[:：])', r'\1\n\2', result)

    # 6. 处理 【标签】 格式
    # 6a. 标签前确保空行分隔（文本开头除外）
    result = re.sub(r'(?<=\S)\n*(【[^】]+】)', r'\n\n\1', result)
    # 6b. 合并标签与被换行拆开的冒号：【tag】\n： → 【tag】：
    result = re.sub(r'(【[^】]+】)\n+([:：])', r'\1\2', result)
    # 6c. 标签后（含可选冒号），如果紧跟非空白非冒号内容则另起一行
    # 用 (?=[^\s:：]) 避免正则回溯将冒号误判为"内容"而拆开 【tag】：
    result = re.sub(r'(【[^】]+】[:：]?)[ \t]*(?=[^\s:：])', r'\1\n', result)

    # 7. 在列表项之间增加视觉空行（排除版本号/小数）
    # 排除 【标签】 行（以】结尾）和子标题行（以冒号结尾）之后的情况，避免标题与首项之间出现空行
    result = re.sub(r'(?<![:：】])\n(\d+\.)(?!\d)', r'\n\n\1', result)

    return result


def _format_standalone_summaries(summaries: dict) -> str:
    """格式化独立展示区概括为纯文本行，每个源名称单独一行"""
    if not summaries:
        return ""
    lines = []
    for source_name, summary in summaries.items():
        if summary:
            lines.append(f"[{source_name}]:\n{summary}")
    return "\n\n".join(lines)


def _format_message_impacts(message_impacts: list) -> str:
    """格式化逐条消息行业映射。"""
    if not message_impacts:
        return ""

    lines = []
    for index, item in enumerate(message_impacts, start=1):
        if not isinstance(item, dict):
            continue

        core_view = item.get("core_view")
        signal_type = item.get("signal_type")
        related_targets = item.get("related_targets")
        direct_chain = item.get("direct_chain")
        indirect_chain = item.get("indirect_chain")

        if core_view or signal_type or related_targets or direct_chain or indirect_chain:
            header_parts = []
            message_id = item.get("id")
            title = item.get("title")
            source = item.get("source")
            industry = item.get("industry")

            if message_id:
                header_parts.append(str(message_id))
            if title:
                header_parts.append(f"「{title}」")
            if source:
                header_parts.append(f"[{source}]")
            if industry:
                header_parts.append(f"主行业：{industry}")

            if header_parts:
                lines.append(f"{index}. " + " | ".join(header_parts))
            if core_view:
                lines.append(f"核心判断：{core_view}")
            if signal_type:
                lines.append(f"催化类型：{signal_type}")
            if related_targets:
                if isinstance(related_targets, (list, tuple, set)):
                    targets_text = "、".join(
                        str(target).strip()
                        for target in related_targets
                        if str(target).strip()
                    )
                else:
                    targets_text = str(related_targets).strip()
                if targets_text:
                    lines.append(f"相关标的：{targets_text}")
            if direct_chain:
                lines.append(f"直接传导：{direct_chain}")
            if indirect_chain:
                lines.append(f"间接传导：{indirect_chain}")
            lines.append("")
            continue

        parts = []
        message_id = item.get("id")
        title = item.get("title")
        source = item.get("source")
        industry = item.get("industry")
        impact = item.get("impact")
        holding_status = item.get("holding_status")
        opportunity_level = item.get("opportunity_level")

        if message_id:
            parts.append(str(message_id))
        if title:
            parts.append(f"「{title}」")
        if source:
            parts.append(f"[{source}]")
        if industry:
            parts.append(str(industry))
        if impact:
            parts.append(str(impact))
        if holding_status:
            parts.append(str(holding_status))
        if opportunity_level:
            parts.append(str(opportunity_level))

        if parts:
            lines.append(f"{index}. " + " | ".join(parts))

    return "\n".join(lines).strip()


def _format_action_items(items: list, is_holding: bool) -> list[str]:
    """格式化评级动作列表。"""
    if not items:
        return []

    lines = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue

        if is_holding:
            code = item.get("code", "")
            name = item.get("name") or code or "未命名持仓"
            target = f"{name}({code})" if code and name != code else name
        else:
            target = item.get("target") or item.get("name") or "未命名标的"

        rating = item.get("rating", "")
        reason = item.get("reason", "")

        line = f"{index}. {target}"
        if rating:
            line += f"：{rating}"
        if reason:
            line += f"；{reason}"
        lines.append(line)

    return lines


def _format_portfolio_summary(portfolio: dict) -> str:
    """
    格式化持仓汇总报告为纯文本

    Args:
        portfolio: 持仓汇总 dict，包含:
            - matrix_distribution: {"黄金区": [...], "需谨慎": [...], "烟蒂区": [...], "双杀区": [...]}
            - top_opportunities: ["候选1", ...]
            - risk_warnings: ["风险提示1", ...]
            - action_suggestions: ["建议1", ...]

    Returns:
        格式化后的文本
    """
    if not portfolio:
        return ""

    raw_content = portfolio.get("raw_content")
    if raw_content:
        return str(raw_content).strip()

    in_portfolio_actions = portfolio.get("in_portfolio_actions", [])
    out_of_portfolio_actions = portfolio.get("out_of_portfolio_actions", [])
    if in_portfolio_actions or out_of_portfolio_actions:
        lines = []

        in_lines = _format_action_items(in_portfolio_actions, is_holding=True)
        if in_lines:
            lines.append("【持仓内】")
            lines.extend(in_lines)
            lines.append("")

        out_lines = _format_action_items(out_of_portfolio_actions, is_holding=False)
        if out_lines:
            lines.append("【持仓外】")
            lines.extend(out_lines)
            lines.append("")

        warnings = portfolio.get("risk_warnings", [])
        if warnings:
            lines.append("【风险提示】")
            for index, warn in enumerate(warnings, start=1):
                lines.append(f"{index}. {warn}")

        return "\n".join(lines).strip()

    lines = []

    # 矩阵分布
    matrix = portfolio.get("matrix_distribution", {})
    if matrix:
        lines.append("【矩阵分布】")
        for zone_key in ["黄金区", "需谨慎", "烟蒂区", "双杀区"]:
            items = matrix.get(zone_key, [])
            if items:
                lines.append(f"{zone_key}: {', '.join(items)}")
        lines.append("")

    # 重点机会
    opportunities = portfolio.get("top_opportunities", [])
    if opportunities:
        lines.append("【优先机会】")
        for index, opp in enumerate(opportunities, start=1):
            lines.append(f"{index}. {opp}")
        lines.append("")

    # 风险提示
    warnings = portfolio.get("risk_warnings", [])
    if warnings:
        lines.append("【风险提示】")
        for index, warn in enumerate(warnings, start=1):
            lines.append(f"{index}. {warn}")
        lines.append("")

    # 操作建议
    suggestions = portfolio.get("action_suggestions", [])
    if suggestions:
        lines.append("【操作建议】")
        for index, sug in enumerate(suggestions, start=1):
            lines.append(f"{index}. {sug}")

    return "\n".join(lines).strip()


def _get_research_sections(result: AIAnalysisResult) -> list[tuple[str, str]]:
    """生成研报式输出的核心 section。"""
    sections = []

    if result.report_overview:
        sections.append(("核心判断", _format_list_content(result.report_overview)))

    key_message_impacts = result.key_message_impacts or result.message_impacts
    if key_message_impacts:
        sections.append(("重点消息", _format_message_impacts(key_message_impacts)))

    portfolio_text = _format_portfolio_summary(result.portfolio_summary)
    if portfolio_text:
        sections.append(("评级建议", portfolio_text))

    return [(title, content) for title, content in sections if content]


def _decorate_feishu_output(content: str) -> str:
    """为飞书 Markdown 输出添加轻量表情装饰。"""
    if not content:
        return ""

    for old, new in _FEISHU_MARKDOWN_REPLACEMENTS.items():
        content = content.replace(old, new)
    return content


def _has_research_sections(result: AIAnalysisResult) -> bool:
    return bool(_get_research_sections(result))


def _render_research_markdown(result: AIAnalysisResult) -> str:
    lines = ["**✨ AI 热点研判**", ""]
    for title, content in _get_research_sections(result):
        lines.extend([f"**{title}**", content, ""])
    return "\n".join(lines)


def _render_research_dingtalk(result: AIAnalysisResult) -> str:
    lines = ["### ✨ AI 热点研判", ""]
    for title, content in _get_research_sections(result):
        lines.extend([f"#### {title}", content, ""])
    return "\n".join(lines)


def _render_research_plain(result: AIAnalysisResult) -> str:
    lines = ["【✨ AI 热点研判】", ""]
    for title, content in _get_research_sections(result):
        lines.extend([f"[{title}]", content, ""])
    return "\n".join(lines)


def _render_research_telegram(result: AIAnalysisResult) -> str:
    lines = ["<b>✨ AI 热点研判</b>", ""]
    for title, content in _get_research_sections(result):
        lines.extend([f"<b>{_escape_html(title)}</b>", _escape_html(content), ""])
    return "\n".join(lines)


def _render_research_html(result: AIAnalysisResult) -> str:
    html_parts = ['<div class="ai-analysis">', "<h3>✨ AI 热点研判</h3>"]
    for title, content in _get_research_sections(result):
        content_html = _escape_html(content).replace("\n", "<br>")
        html_parts.extend(
            [
                '<div class="ai-section">',
                f"<h4>{_escape_html(title)}</h4>",
                f'<div class="ai-content">{content_html}</div>',
                "</div>",
            ]
        )
    html_parts.append("</div>")
    return "\n".join(html_parts)


def _render_research_html_rich(result: AIAnalysisResult) -> str:
    ai_html = """
                <div class="ai-section">
                    <div class="ai-section-header">
                        <div class="ai-section-title">✨ AI 热点研判</div>
                        <span class="ai-section-badge">AI</span>
                    </div>
                    <div class="ai-blocks-grid">"""

    for title, content in _get_research_sections(result):
        content_html = _escape_html(content).replace("\n", "<br>")
        ai_html += f"""
                    <div class="ai-block">
                        <div class="ai-block-title">{_escape_html(title)}</div>
                        <div class="ai-block-content">{content_html}</div>
                    </div>"""

    ai_html += """
                    </div>
                </div>"""
    return ai_html


def render_ai_analysis_markdown(result: AIAnalysisResult) -> str:
    """渲染为通用 Markdown 格式（Telegram、企业微信、ntfy、Bark、Slack）"""
    if not result.success:
        if result.skipped:
            return f"ℹ️ {result.error}"
        return f"⚠️ AI 分析失败: {result.error}"

    if _has_research_sections(result):
        return _render_research_markdown(result)

    lines = ["**✨ AI 热点分析**", ""]

    if result.message_impacts:
        lines.extend(["**消息行业映射**", _format_message_impacts(result.message_impacts), ""])

    if result.personal_layer:
        lines.extend(["**个人层**", _format_list_content(result.personal_layer), ""])

    if result.regional_layer:
        lines.extend(
            ["**地区层**", _format_list_content(result.regional_layer), ""]
        )

    if result.social_layer:
        lines.extend(["**行业层**", _format_list_content(result.social_layer), ""])

    if result.national_layer:
        lines.extend(
            ["**国家层**", _format_list_content(result.national_layer), ""]
        )

    if result.tech_layer:
        lines.extend(["**科技层**", _format_list_content(result.tech_layer), ""])

    if result.portfolio_summary:
        portfolio_text = _format_portfolio_summary(result.portfolio_summary)
        if portfolio_text:
            lines.extend(["**持仓汇总**", portfolio_text, ""])

    return "\n".join(lines)


def render_ai_analysis_feishu(result: AIAnalysisResult) -> str:
    """渲染为飞书卡片 Markdown 格式"""
    if not result.success:
        if result.skipped:
            return f"ℹ️ {result.error}"
        return f"⚠️ AI 分析失败: {result.error}"

    if _has_research_sections(result):
        return _decorate_feishu_output(_render_research_markdown(result))

    lines = ["**✨ AI 热点分析**", ""]

    if result.message_impacts:
        lines.extend(["**消息行业映射**", _format_message_impacts(result.message_impacts), ""])

    if result.personal_layer:
        lines.extend(["**个人层**", _format_list_content(result.personal_layer), ""])

    if result.regional_layer:
        lines.extend(
            ["**地区层**", _format_list_content(result.regional_layer), ""]
        )

    if result.social_layer:
        lines.extend(["**行业层**", _format_list_content(result.social_layer), ""])

    if result.national_layer:
        lines.extend(
            ["**国家层**", _format_list_content(result.national_layer), ""]
        )

    if result.tech_layer:
        lines.extend(["**科技层**", _format_list_content(result.tech_layer), ""])

    if result.portfolio_summary:
        portfolio_text = _format_portfolio_summary(result.portfolio_summary)
        if portfolio_text:
            lines.extend(["**持仓汇总**", portfolio_text, ""])

    return _decorate_feishu_output("\n".join(lines))


def render_ai_analysis_dingtalk(result: AIAnalysisResult) -> str:
    """渲染为钉钉 Markdown 格式"""
    if not result.success:
        if result.skipped:
            return f"ℹ️ {result.error}"
        return f"⚠️ AI 分析失败: {result.error}"

    if _has_research_sections(result):
        return _render_research_dingtalk(result)

    lines = ["### ✨ AI 热点分析", ""]

    if result.message_impacts:
        lines.extend(["#### 消息行业映射", _format_message_impacts(result.message_impacts), ""])

    if result.personal_layer:
        lines.extend(
            ["#### 个人层", _format_list_content(result.personal_layer), ""]
        )

    if result.regional_layer:
        lines.extend(
            [
                "#### 地区层",
                _format_list_content(result.regional_layer),
                "",
            ]
        )

    if result.social_layer:
        lines.extend(["#### 行业层", _format_list_content(result.social_layer), ""])

    if result.national_layer:
        lines.extend(
            ["#### 国家层", _format_list_content(result.national_layer), ""]
        )

    if result.tech_layer:
        lines.extend(["#### 科技层", _format_list_content(result.tech_layer), ""])

    if result.portfolio_summary:
        portfolio_text = _format_portfolio_summary(result.portfolio_summary)
        if portfolio_text:
            lines.extend(["#### 持仓汇总", portfolio_text])

    return "\n".join(lines)


def render_ai_analysis_html(result: AIAnalysisResult) -> str:
    """渲染为 HTML 格式（邮件）"""
    if not result.success:
        if result.skipped:
            return f'<div class="ai-info">ℹ️ {_escape_html(result.error)}</div>'
        return (
            f'<div class="ai-error">⚠️ AI 分析失败: {_escape_html(result.error)}</div>'
        )

    if _has_research_sections(result):
        return _render_research_html(result)

    html_parts = ['<div class="ai-analysis">', "<h3>✨ AI 热点分析</h3>"]

    if result.message_impacts:
        content = _format_message_impacts(result.message_impacts)
        content_html = _escape_html(content).replace("\n", "<br>")
        html_parts.extend(
            [
                '<div class="ai-section">',
                "<h4>消息行业映射</h4>",
                f'<div class="ai-content">{content_html}</div>',
                "</div>",
            ]
        )

    if result.personal_layer:
        content = _format_list_content(result.personal_layer)
        content_html = _escape_html(content).replace("\n", "<br>")
        html_parts.extend(
            [
                '<div class="ai-section">',
                "<h4>个人层</h4>",
                f'<div class="ai-content">{content_html}</div>',
                "</div>",
            ]
        )

    if result.regional_layer:
        content = _format_list_content(result.regional_layer)
        content_html = _escape_html(content).replace("\n", "<br>")
        html_parts.extend(
            [
                '<div class="ai-section">',
                "<h4>地区层</h4>",
                f'<div class="ai-content">{content_html}</div>',
                "</div>",
            ]
        )

    if result.social_layer:
        content = _format_list_content(result.social_layer)
        content_html = _escape_html(content).replace("\n", "<br>")
        html_parts.extend(
            [
                '<div class="ai-section">',
                "<h4>行业层</h4>",
                f'<div class="ai-content">{content_html}</div>',
                "</div>",
            ]
        )

    if result.national_layer:
        content = _format_list_content(result.national_layer)
        content_html = _escape_html(content).replace("\n", "<br>")
        html_parts.extend(
            [
                '<div class="ai-section">',
                "<h4>国家层</h4>",
                f'<div class="ai-content">{content_html}</div>',
                "</div>",
            ]
        )

    if result.tech_layer:
        content = _format_list_content(result.tech_layer)
        content_html = _escape_html(content).replace("\n", "<br>")
        html_parts.extend(
            [
                '<div class="ai-section">',
                "<h4>科技层</h4>",
                f'<div class="ai-content">{content_html}</div>',
                "</div>",
            ]
        )

    if result.portfolio_summary:
        portfolio_text = _format_portfolio_summary(result.portfolio_summary)
        if portfolio_text:
            portfolio_html = _escape_html(portfolio_text).replace("\n", "<br>")
            html_parts.extend(
                [
                    '<div class="ai-section ai-conclusion">',
                    "<h4>持仓汇总</h4>",
                    f'<div class="ai-content">{portfolio_html}</div>',
                    "</div>",
                ]
            )

    html_parts.append("</div>")
    return "\n".join(html_parts)


def render_ai_analysis_plain(result: AIAnalysisResult) -> str:
    """渲染为纯文本格式"""
    if not result.success:
        if result.skipped:
            return result.error
        return f"AI 分析失败: {result.error}"

    if _has_research_sections(result):
        return _render_research_plain(result)

    lines = ["【✨ AI 热点分析】", ""]

    if result.message_impacts:
        lines.extend(["[消息行业映射]", _format_message_impacts(result.message_impacts), ""])

    if result.personal_layer:
        lines.extend(["[个人层]", _format_list_content(result.personal_layer), ""])

    if result.regional_layer:
        lines.extend(
            ["[地区层]", _format_list_content(result.regional_layer), ""]
        )

    if result.social_layer:
        lines.extend(["[行业层]", _format_list_content(result.social_layer), ""])

    if result.national_layer:
        lines.extend(["[国家层]", _format_list_content(result.national_layer), ""])

    if result.tech_layer:
        lines.extend(["[科技层]", _format_list_content(result.tech_layer), ""])

    if result.portfolio_summary:
        portfolio_text = _format_portfolio_summary(result.portfolio_summary)
        if portfolio_text:
            lines.extend(["[持仓汇总]", portfolio_text])

    return "\n".join(lines)


def render_ai_analysis_telegram(result: AIAnalysisResult) -> str:
    """渲染为 Telegram HTML 格式（配合 parse_mode: HTML）

    Telegram Bot API 的 HTML 模式仅支持有限标签：
    <b>, <i>, <u>, <s>, <code>, <pre>, <a href="">, <blockquote>
    换行直接使用 \\n，不支持 <br>, <div>, <h1>-<h6> 等标签。
    """
    if not result.success:
        if result.skipped:
            return f"ℹ️ {_escape_html(result.error)}"
        return f"⚠️ AI 分析失败: {_escape_html(result.error)}"

    if _has_research_sections(result):
        return _render_research_telegram(result)

    lines = ["<b>✨ AI 热点分析</b>", ""]

    if result.message_impacts:
        lines.extend(["<b>消息行业映射</b>", _escape_html(_format_message_impacts(result.message_impacts)), ""])

    if result.personal_layer:
        lines.extend(["<b>个人层</b>", _escape_html(_format_list_content(result.personal_layer)), ""])

    if result.regional_layer:
        lines.extend(["<b>地区层</b>", _escape_html(_format_list_content(result.regional_layer)), ""])

    if result.social_layer:
        lines.extend(["<b>行业层</b>", _escape_html(_format_list_content(result.social_layer)), ""])

    if result.national_layer:
        lines.extend(["<b>国家层</b>", _escape_html(_format_list_content(result.national_layer)), ""])

    if result.tech_layer:
        lines.extend(["<b>科技层</b>", _escape_html(_format_list_content(result.tech_layer)), ""])

    if result.portfolio_summary:
        portfolio_text = _format_portfolio_summary(result.portfolio_summary)
        if portfolio_text:
            lines.extend(["<b>持仓汇总</b>", _escape_html(portfolio_text)])

    return "\n".join(lines)


def get_ai_analysis_renderer(channel: str):
    """根据渠道获取对应的渲染函数"""
    renderers = {
        "feishu": render_ai_analysis_feishu,
        "dingtalk": render_ai_analysis_dingtalk,
        "wework": render_ai_analysis_markdown,
        "telegram": render_ai_analysis_telegram,
        "email": render_ai_analysis_html_rich,  # 邮件使用丰富样式，配合 HTML 报告的 CSS
        "ntfy": render_ai_analysis_markdown,
        "bark": render_ai_analysis_plain,
        "slack": render_ai_analysis_markdown,
    }
    return renderers.get(channel, render_ai_analysis_markdown)


def render_ai_analysis_html_rich(result: AIAnalysisResult) -> str:
    """渲染为丰富样式的 HTML 格式（HTML 报告用）"""
    if not result:
        return ""

    # 检查是否成功
    if not result.success:
        if result.skipped:
            return f"""
                <div class="ai-section">
                    <div class="ai-info">ℹ️ {_escape_html(str(result.error))}</div>
                </div>"""
        error_msg = result.error or "未知错误"
        return f"""
                <div class="ai-section">
                    <div class="ai-error">⚠️ AI 分析失败: {_escape_html(str(error_msg))}</div>
                </div>"""

    if _has_research_sections(result):
        return _render_research_html_rich(result)

    ai_html = """
                <div class="ai-section">
                    <div class="ai-section-header">
                        <div class="ai-section-title">✨ AI 热点分析</div>
                        <span class="ai-section-badge">AI</span>
                    </div>
                    <div class="ai-blocks-grid">"""

    if result.message_impacts:
        content = _format_message_impacts(result.message_impacts)
        content_html = _escape_html(content).replace("\n", "<br>")
        ai_html += f"""
                    <div class="ai-block">
                        <div class="ai-block-title">消息行业映射</div>
                        <div class="ai-block-content">{content_html}</div>
                    </div>"""

    if result.personal_layer:
        content = _format_list_content(result.personal_layer)
        content_html = _escape_html(content).replace("\n", "<br>")
        ai_html += f"""
                    <div class="ai-block">
                        <div class="ai-block-title">个人层</div>
                        <div class="ai-block-content">{content_html}</div>
                    </div>"""

    if result.regional_layer:
        content = _format_list_content(result.regional_layer)
        content_html = _escape_html(content).replace("\n", "<br>")
        ai_html += f"""
                    <div class="ai-block">
                        <div class="ai-block-title">地区层</div>
                        <div class="ai-block-content">{content_html}</div>
                    </div>"""

    if result.social_layer:
        content = _format_list_content(result.social_layer)
        content_html = _escape_html(content).replace("\n", "<br>")
        ai_html += f"""
                    <div class="ai-block">
                        <div class="ai-block-title">行业层</div>
                        <div class="ai-block-content">{content_html}</div>
                    </div>"""

    if result.national_layer:
        content = _format_list_content(result.national_layer)
        content_html = _escape_html(content).replace("\n", "<br>")
        ai_html += f"""
                    <div class="ai-block">
                        <div class="ai-block-title">国家层</div>
                        <div class="ai-block-content">{content_html}</div>
                    </div>"""

    if result.tech_layer:
        content = _format_list_content(result.tech_layer)
        content_html = _escape_html(content).replace("\n", "<br>")
        ai_html += f"""
                    <div class="ai-block">
                        <div class="ai-block-title">科技层</div>
                        <div class="ai-block-content">{content_html}</div>
                    </div>"""

    if result.portfolio_summary:
        portfolio_text = _format_portfolio_summary(result.portfolio_summary)
        if portfolio_text:
            portfolio_html = _escape_html(portfolio_text).replace("\n", "<br>")
            ai_html += f"""
                    <div class="ai-block">
                        <div class="ai-block-title">持仓汇总</div>
                        <div class="ai-block-content">{portfolio_html}</div>
                    </div>"""

    ai_html += """
                    </div>
                </div>"""
    return ai_html
