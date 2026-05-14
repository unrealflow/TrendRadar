# coding=utf-8
"""
东方财富妙想(Miaoxiang) 数据接口封装

基于妙想 Skill 包中的 mx-data（行情查询）和 mx-search（资讯搜索）实现，
为 TrendRadar AI 分析提供权威金融数据补充。

API 文档参考:
- mx-data: https://marketing.dfcfw.com/res/download/A620260331IHX67H.zip
- mx-search: https://marketing.dfcfw.com/res/download/A620260331K5WDTK.zip
"""

import json
import os
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MX_DATA_URL = "https://mkapi2.dfcfs.com/finskillshub/api/claw/query"
MX_SEARCH_URL = "https://mkapi2.dfcfs.com/finskillshub/api/claw/news-search"
DEFAULT_TIMEOUT = 30.0

# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------

class MiaoxiangClient:
    """妙想数据客户端"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MX_APIKEY", "")

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "apikey": self.api_key,
        }

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(url, headers=self._headers(), json=payload, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # 行情 / 财务数据查询 (mx-data)
    # ------------------------------------------------------------------

    def query_data(self, tool_query: str) -> Dict[str, Any]:
        """
        查询金融数据（行情、财务、基本面等）

        Args:
            tool_query: 自然语言查询，例如 "沪深300最新点位"、"贵州茅台收盘价"

        Returns:
            API 原始响应 JSON
        """
        return self._post(MX_DATA_URL, {"toolQuery": tool_query})

    # ------------------------------------------------------------------
    # 资讯搜索 (mx-search)
    # ------------------------------------------------------------------

    def search_news(self, query: str) -> Dict[str, Any]:
        """
        搜索金融资讯（新闻、公告、研报等）

        Args:
            query: 搜索问句，例如 "人工智能板块最新新闻"、"美联储降息影响"

        Returns:
            API 原始响应 JSON
        """
        return self._post(MX_SEARCH_URL, {"query": query})


# ---------------------------------------------------------------------------
# 结果解析与格式化（供 AI prompt 使用）
# ---------------------------------------------------------------------------

def _safe_get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """安全多级字典取值"""
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key, default)
        if data is None:
            return default
    return data


def format_data_query_result(result: Dict[str, Any]) -> str:
    """
    将 mx-data 查询结果格式化为 prompt 可用文本。
    提取数据表中的关键指标，按证券分组展示。
    """
    status = result.get("status")
    message = result.get("message", "")
    if status != 0:
        return f"[妙想行情] 查询失败: {message} (status={status})"

    inner_data = _safe_get(result, "data", "data", default={})
    search_result = inner_data.get("searchDataResultDTO", {})
    dto_list = search_result.get("dataTableDTOList", [])
    entity_tags = search_result.get("entityTagDTOList", [])

    if not dto_list:
        return "[妙想行情] 未返回有效数据表"

    lines: List[str] = []

    # 顶部列出涉及的证券
    if entity_tags:
        entities = []
        for tag in entity_tags:
            name = tag.get("fullName", "")
            code = tag.get("secuCode", "")
            type_name = tag.get("entityTypeName", "")
            parts = [p for p in [name, code, type_name] if p]
            if parts:
                entities.append(" / ".join(parts))
        if entities:
            lines.append("【涉及证券】" + "；".join(entities))

    # 逐表展示指标数据
    for dto in dto_list:
        if not isinstance(dto, dict):
            continue

        title = dto.get("title") or dto.get("entityName") or "未命名指标"
        lines.append(f"\n--- {title} ---")

        # condition 说明
        condition = dto.get("condition")
        if condition:
            lines.append(f"条件: {condition}")

        # 表格数据
        table = dto.get("table", {})
        name_map = dto.get("nameMap", {})
        if isinstance(name_map, list):
            name_map = {str(i): v for i, v in enumerate(name_map)}
        elif not isinstance(name_map, dict):
            name_map = {}

        headers = table.get("headName", [])
        if not isinstance(headers, list):
            headers = []

        # 收集指标列（排除 headName）
        indicator_keys = [k for k in table.keys() if k != "headName"]

        if len(headers) > 0 and indicator_keys:
            # 多日期格式：每行一个日期
            fieldnames = ["日期"] + [str(name_map.get(k, k)) for k in indicator_keys]
            lines.append(" | ".join(fieldnames))
            for row_idx, date in enumerate(headers):
                cells = [str(date)]
                for key in indicator_keys:
                    raw_values = table.get(key, [])
                    val = raw_values[row_idx] if row_idx < len(raw_values) else ""
                    cells.append(str(val))
                lines.append(" | ".join(cells))
        elif len(headers) == 1 and indicator_keys:
            # 单日期（如最新行情）
            for key in indicator_keys:
                raw_values = table.get(key, [])
                val = raw_values[0] if isinstance(raw_values, list) and raw_values else raw_values
                label = str(name_map.get(key, key))
                lines.append(f"{label}: {val}")
        else:
            lines.append("（无表格数据）")

    return "\n".join(lines)


def format_news_search_result(result: Dict[str, Any]) -> str:
    """
    将 mx-search 查询结果格式化为 prompt 可用文本。
    提取资讯列表，按类型分组展示。
    """
    status = result.get("status")
    message = result.get("message", "")
    if status != 0:
        return f"[妙想资讯] 查询失败: {message} (status={status})"

    inner_data = _safe_get(result, "data", "data", default={})
    search_response = inner_data.get("llmSearchResponse", {})
    items = search_response.get("data", [])

    if not items:
        return "[妙想资讯] 未找到相关资讯"

    lines: List[str] = []
    lines.append(f"【妙想资讯】共 {len(items)} 条相关资讯:\n")

    type_map = {
        "REPORT": "研报",
        "NEWS": "新闻",
        "ANNOUNCEMENT": "公告",
    }

    for i, item in enumerate(items, 1):
        title = item.get("title", "无标题")
        content = item.get("content", "")
        date = item.get("date", "")
        ins_name = item.get("insName", "")
        info_type = item.get("informationType", "")
        rating = item.get("rating", "")
        entity_name = item.get("entityFullName", "")

        type_cn = type_map.get(info_type, info_type)

        lines.append(f"{i}. {title}")
        meta = []
        if entity_name:
            meta.append(f"证券: {entity_name}")
        if ins_name:
            meta.append(f"来源: {ins_name}")
        if date:
            meta.append(f"日期: {date.split()[0] if ' ' in date else date}")
        if type_cn:
            meta.append(f"类型: {type_cn}")
        if rating:
            meta.append(f"评级: {rating}")
        if meta:
            lines.append("   " + " | ".join(meta))
        if content:
            # 截断过长内容，避免 prompt 爆炸
            content_preview = content[:500] + ("..." if len(content) > 500 else "")
            lines.append(f"   摘要: {content_preview}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 批量查询辅助（用于持仓快照）
# ---------------------------------------------------------------------------

def build_miaoxiang_market_text(
    symbols: List[str],
    api_key: Optional[str] = None,
) -> str:
    """
    批量查询多个标的的最新行情，返回 prompt 文本。

    Args:
        symbols: 标的代码列表，如 ["300059.SZ", "600519.SH"]
        api_key: 可选 API Key，默认从环境变量读取

    Returns:
        格式化后的行情文本
    """
    client = MiaoxiangClient(api_key=api_key)
    lines: List[str] = []

    for symbol in symbols:
        try:
            result = client.query_data(f"{symbol} 最新价 涨跌幅")
            text = format_data_query_result(result)
            lines.append(text)
        except Exception as e:
            lines.append(f"[妙想行情] {symbol} 查询异常: {e}")

    return "\n\n".join(lines) if lines else "[妙想行情] 无查询结果"


def build_miaoxiang_news_text(
    keywords: List[str],
    api_key: Optional[str] = None,
) -> str:
    """
    批量搜索多个关键词的最新资讯，返回 prompt 文本。

    Args:
        keywords: 关键词列表，如 ["人工智能", "美联储降息"]
        api_key: 可选 API Key，默认从环境变量读取

    Returns:
        格式化后的资讯文本
    """
    client = MiaoxiangClient(api_key=api_key)
    lines: List[str] = []

    for kw in keywords:
        try:
            result = client.search_news(f"{kw} 最新新闻")
            text = format_news_search_result(result)
            lines.append(text)
        except Exception as e:
            lines.append(f"[妙想资讯] {kw} 查询异常: {e}")

    return "\n\n".join(lines) if lines else "[妙想资讯] 无查询结果"


# ---------------------------------------------------------------------------
# 顶层便捷函数（供 analyzer 调用）
# ---------------------------------------------------------------------------

def fetch_miaoxiang_market_snapshot(
    holdings: List[Dict[str, Any]],
    api_key: Optional[str] = None,
) -> str:
    """
    为持仓列表生成妙想行情快照文本。

    从 holdings 中提取名称/代码，向妙想查询最新行情数据，
    返回可直接嵌入 prompt 的文本。
    """
    if not holdings:
        print("[妙想行情] 持仓列表为空，跳过查询")
        return "[妙想行情] 持仓列表为空"

    names: List[str] = []
    for h in holdings:
        name = h.get("name", "")
        code = h.get("code", "")
        if name:
            names.append(name)
        elif code:
            names.append(code)

    if not names:
        print("[妙想行情] 无可查询标的")
        return "[妙想行情] 无可查询标的"

    # 合并为少量查询，避免调用次数过多
    # 妙想 API 支持自然语言，可以一次查多个标的
    merged_names = " ".join(names[:10])
    client = MiaoxiangClient(api_key=api_key)
    try:
        print(f"[妙想行情] 查询标的: {merged_names}")
        result = client.query_data(f"{merged_names} 最新价 涨跌幅")
        text = format_data_query_result(result)
        print(f"[妙想行情] 查询成功，返回 {len(text)} 字符")
        return text
    except Exception as e:
        print(f"[妙想行情] 查询失败: {e}")
        return f"[妙想行情] 查询失败: {e}"


def fetch_miaoxiang_news_snapshot(
    keywords: List[str],
    api_key: Optional[str] = None,
) -> str:
    """
    基于关键词生成妙想资讯快照文本。

    Args:
        keywords: 热榜关键词列表
        api_key: 可选 API Key

    Returns:
        可直接嵌入 prompt 的资讯文本
    """
    if not keywords:
        print("[妙想资讯] 无关键词，跳过查询")
        return "[妙想资讯] 无关键词"

    merged_kws = " ".join(keywords[:5])
    client = MiaoxiangClient(api_key=api_key)
    try:
        print(f"[妙想资讯] 查询关键词: {merged_kws}")
        result = client.search_news(f"{merged_kws} 最新资讯")
        text = format_news_search_result(result)
        print(f"[妙想资讯] 查询成功，返回 {len(text)} 字符")
        return text
    except Exception as e:
        print(f"[妙想资讯] 查询失败: {e}")
        return f"[妙想资讯] 查询失败: {e}"
