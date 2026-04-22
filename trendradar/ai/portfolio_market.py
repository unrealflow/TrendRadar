# coding=utf-8

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests

from trendradar.ai.market_data import fetch_symbol_quote


PORTFOLIO_PROXY_MAP: Dict[str, Dict[str, Any]] = {
    "CITU-BOND": {"symbol": "511220.SS", "label": "城投ETF"},
    "POLICY-BOND": {
        "label": "政策债组合",
        "components": [
            {"symbol": "511090.SS", "weight": 0.365},
            {"symbol": "511160.SS", "weight": 0.635},
        ],
    },
    "CSI-DIVIDEND": {"symbol": "515080.SS", "label": "中证红利ETF"},
    "HSI-HDIV-LOWVOL": {"symbol": "159545.SZ", "label": "港股红利ETF"},
    "NF-EW-SELECT": {"symbol": "3441.HK", "label": "南方东西精选"},
    "GX-VALUE": {"symbol": "512040.SS", "label": "国信价值ETF"},
    "SPX": {"symbol": "513650.SS", "label": "标普ETF"},
    "N225": {"symbol": "513000.SS", "label": "225ETF"},
    "DAX": {"symbol": "159561.SZ", "label": "德国ETF"},
    "APAC-SELECT": {"symbol": "159687.SZ", "label": "亚太精选ETF"},
    "HK-CONNECT-50": {"symbol": "513550.SS", "label": "港股通50ETF"},
    "HSTECH": {"symbol": "513010.SS", "label": "港股科技ETF"},
    "STAR-CHIP-DESIGN": {"symbol": "588780.SS", "label": "芯片设计ETF"},
    "STAR-VALUE": {"symbol": "588910.SS", "label": "科创价值ETF"},
    "GOLD": {"symbol": "518850.SS", "label": "黄金9999ETF"},
    "SOYMEAL": {"symbol": "159985.SZ", "label": "豆粕ETF"},
    "BASE-METALS": {"symbol": "159980.SZ", "label": "有色ETF"},
    "OIL-GAS": {"symbol": "162411.SZ", "label": "华宝油气"},
    "SE-ASIA-TECH": {"symbol": "513730.SS", "label": "东南亚ETF"},
    "HK-INNOV-DRUG": {"symbol": "513120.SS", "label": "港股创新药ETF"},
    "GRID-EQUIP": {"symbol": "159326.SZ", "label": "电网设备ETF"},
}


def _format_pct(value: float) -> str:
    return f"{value:+.2f}%"


def _format_pct_point(value: float) -> str:
    return f"{value:+.2f}pct"


def _fetch_symbol_quote(symbol: str, timeout: float) -> Optional[Dict[str, Any]]:
    return fetch_symbol_quote(symbol, timeout, request_get=requests.get)


def _collect_symbols(holdings: List[Dict[str, Any]]) -> List[str]:
    symbols = set()
    for holding in holdings:
        mapping = PORTFOLIO_PROXY_MAP.get(holding.get("code", ""))
        if not mapping:
            continue
        if "components" in mapping:
            for component in mapping["components"]:
                symbol = component.get("symbol")
                if symbol:
                    symbols.add(symbol)
            continue
        symbol = mapping.get("symbol")
        if symbol:
            symbols.add(symbol)
    return sorted(symbols)


def _fetch_quotes(symbols: List[str], timeout: float) -> Dict[str, Optional[Dict[str, Any]]]:
    if not symbols:
        return {}

    quotes: Dict[str, Optional[Dict[str, Any]]] = {}
    max_workers = min(6, len(symbols))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_fetch_symbol_quote, symbol, timeout): symbol
            for symbol in symbols
        }
        for future in as_completed(future_map):
            symbol = future_map[future]
            try:
                quotes[symbol] = future.result()
            except Exception:
                quotes[symbol] = None
    return quotes


def _resolve_holding_quote(
    holding: Dict[str, Any],
    quotes: Dict[str, Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    code = holding.get("code", "")
    mapping = PORTFOLIO_PROXY_MAP.get(code)
    if not mapping:
        return None

    if "components" in mapping:
        total_weight = 0.0
        pct_change = 0.0
        labels = []
        for component in mapping["components"]:
            symbol = component.get("symbol")
            component_weight = float(component.get("weight", 0.0))
            quote_data = quotes.get(symbol)
            if not quote_data or component_weight <= 0:
                continue
            pct_change += quote_data["pct_change"] * component_weight
            total_weight += component_weight
            labels.append(quote_data.get("label") or symbol)

        if total_weight <= 0:
            return None

        pct_change /= total_weight
        proxy_label = mapping.get("label") or "/".join(labels)
    else:
        symbol = mapping.get("symbol")
        quote_data = quotes.get(symbol)
        if not quote_data:
            return None
        pct_change = quote_data["pct_change"]
        proxy_label = mapping.get("label") or quote_data.get("label") or symbol

    weight_pct = float(holding.get("weight_pct", 0.0))
    contribution_pct = pct_change * weight_pct / 100.0
    return {
        "code": code,
        "name": holding.get("name") or code,
        "weight_pct": weight_pct,
        "proxy_label": proxy_label,
        "pct_change": pct_change,
        "contribution_pct": contribution_pct,
    }


def _build_market_summary(
    covered_positions: List[Dict[str, Any]],
    uncovered_weight_pct: float,
) -> str:
    positive = [item for item in covered_positions if item["contribution_pct"] > 0]
    negative = [item for item in covered_positions if item["contribution_pct"] < 0]

    positive.sort(key=lambda item: item["contribution_pct"], reverse=True)
    negative.sort(key=lambda item: item["contribution_pct"])

    parts = []
    if positive:
        parts.append(f"支撑方向集中在{'、'.join(item['name'] for item in positive[:2])}")
    if negative:
        parts.append(f"回撤主要来自{'、'.join(item['name'] for item in negative[:2])}")
    if not parts:
        parts.append("覆盖仓位整体波动有限")
    if uncovered_weight_pct > 0:
        parts.append(f"另有 {uncovered_weight_pct:.0f}% 仓位未纳入代理行情")
    return "；".join(parts) + "。"


def _build_holding_notes(covered_positions: List[Dict[str, Any]]) -> List[str]:
    ranked = sorted(
        covered_positions,
        key=lambda item: abs(item["contribution_pct"]),
        reverse=True,
    )
    notes = []
    for item in ranked[:4]:
        contribution_text = _format_pct_point(item["contribution_pct"])
        move_text = _format_pct(item["pct_change"])
        notes.append(
            f"{item['name']}：代理涨跌 {move_text}，单日贡献 {contribution_text}。"
        )
    return notes


def _build_prompt_text(snapshot: Dict[str, Any]) -> str:
    lines = [
        "以下为默认持仓的最新可用代理行情快照。不同市场交易时区不同，只能用于估算当日表现，不得外推为精确净值。",
        f"- 代理组合估算收益：{_format_pct(snapshot['estimated_return_pct'])}",
        f"- 已覆盖仓位：{snapshot['covered_weight_pct']:.0f}%",
        f"- 未覆盖仓位：{snapshot['uncovered_weight_pct']:.0f}%",
        f"- 盘面摘要：{snapshot['market_summary']}",
    ]

    if snapshot["top_positive"]:
        support_text = "、".join(
            f"{item['name']}({_format_pct_point(item['contribution_pct'])})"
            for item in snapshot["top_positive"]
        )
        lines.append(f"- 主要支撑：{support_text}")
    if snapshot["top_negative"]:
        drag_text = "、".join(
            f"{item['name']}({_format_pct_point(item['contribution_pct'])})"
            for item in snapshot["top_negative"]
        )
        lines.append(f"- 主要拖累：{drag_text}")

    if snapshot["holding_notes"]:
        lines.append("- 持仓观察：")
        lines.extend(f"  - {note}" for note in snapshot["holding_notes"])

    return "\n".join(lines)


def build_portfolio_market_snapshot(
    holdings: List[Dict[str, Any]],
    timeout: float = 3.0,
) -> Dict[str, Any]:
    if not holdings:
        return {
            "prompt_text": "默认持仓为空，无法生成代理行情快照。",
        }

    quotes = _fetch_quotes(_collect_symbols(holdings), timeout)
    covered_positions = []
    uncovered_positions = []

    for holding in holdings:
        resolved = _resolve_holding_quote(holding, quotes)
        if resolved:
            covered_positions.append(resolved)
        else:
            uncovered_positions.append(holding)

    if not covered_positions:
        return {
            "prompt_text": "代理行情快照暂不可用，请不要编造当日收益数字，只做定性分析。",
        }

    covered_weight_pct = sum(item["weight_pct"] for item in covered_positions)
    uncovered_weight_pct = max(0.0, 100.0 - covered_weight_pct)
    estimated_return_pct = sum(item["contribution_pct"] for item in covered_positions)

    ranked_positive = sorted(
        [item for item in covered_positions if item["contribution_pct"] > 0],
        key=lambda item: item["contribution_pct"],
        reverse=True,
    )
    ranked_negative = sorted(
        [item for item in covered_positions if item["contribution_pct"] < 0],
        key=lambda item: item["contribution_pct"],
    )

    market_summary = _build_market_summary(covered_positions, uncovered_weight_pct)
    holding_notes = _build_holding_notes(covered_positions)
    summary = (
        f"按最新可用交易时段估算，代理组合收益 {_format_pct(estimated_return_pct)}，"
        f"已覆盖仓位 {covered_weight_pct:.0f}%。"
    )
    if ranked_positive:
        summary += f" 主要支撑来自{'、'.join(item['name'] for item in ranked_positive[:2])}。"
    if ranked_negative:
        summary += f" 主要拖累来自{'、'.join(item['name'] for item in ranked_negative[:2])}。"
    if uncovered_weight_pct > 0:
        summary += f" 另有 {uncovered_weight_pct:.0f}% 仓位未覆盖。"

    snapshot = {
        "estimated_return_pct": round(estimated_return_pct, 2),
        "covered_weight_pct": round(covered_weight_pct, 1),
        "uncovered_weight_pct": round(uncovered_weight_pct, 1),
        "market_summary": market_summary,
        "holding_notes": holding_notes,
        "top_positive": ranked_positive[:2],
        "top_negative": ranked_negative[:2],
        "daily_performance": {
            "summary": summary,
            "estimated_return_pct": round(estimated_return_pct, 2),
            "covered_weight_pct": round(covered_weight_pct, 1),
            "uncovered_weight_pct": round(uncovered_weight_pct, 1),
        },
    }
    snapshot["prompt_text"] = _build_prompt_text(snapshot)
    return snapshot