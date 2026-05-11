# coding=utf-8

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests

from trendradar.ai.market_data import fetch_symbol_history
from trendradar.ai.portfolio_market import PORTFOLIO_PROXY_MAP

TREND_SCORES = {
    "道氏上升趋势延续": 2,
    "道氏上升趋势": 2,
    "趋势转强": 1,
    "震荡整理": 0,
    "趋势转弱": -1,
    "道氏下降趋势": -2,
    "道氏下降趋势延续": -2,
}

COMPOSITE_TREND_SCORES = {
    "确立多头": 2,
    "转强中": 1,
    "震荡中": 0,
    "转弱中": -1,
    "确立空头": -2,
}


def _fetch_symbol_history(symbol: str, timeout: float) -> Optional[Dict[str, Any]]:
    return fetch_symbol_history(symbol, timeout, request_get=requests.get)


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


def _fetch_histories(symbols: List[str], timeout: float) -> Dict[str, Optional[Dict[str, Any]]]:
    if not symbols:
        return {}

    histories: Dict[str, Optional[Dict[str, Any]]] = {}
    max_workers = min(6, len(symbols))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_fetch_symbol_history, symbol, timeout): symbol
            for symbol in symbols
        }
        for future in as_completed(future_map):
            symbol = future_map[future]
            try:
                histories[symbol] = future.result()
            except Exception:
                histories[symbol] = None
    return histories


def _combine_component_histories(
    components: List[Dict[str, Any]],
    histories: Dict[str, Any],
) -> Optional[List[Dict[str, float]]]:
    valid_components = []
    for component in components:
        symbol = component.get("symbol")
        weight = float(component.get("weight", 0.0))
        history = histories.get(symbol)
        if weight <= 0 or not history:
            continue
        valid_components.append((history["series"], weight))

    if not valid_components:
        return None

    total_weight = sum(weight for _, weight in valid_components)
    min_len = min(len(series) for series, _ in valid_components)
    if total_weight <= 0 or min_len < 60:
        return None

    combined = []
    for index in range(min_len):
        close = 0.0
        high = 0.0
        low = 0.0
        date_text = ""
        for series, weight in valid_components:
            point = series[-min_len + index]
            close += point["close"] * weight
            high += point["high"] * weight
            low += point["low"] * weight
            if not date_text:
                date_text = str(point.get("date", "")).strip()
        combined.append(
            {
                "date": date_text,
                "close": close / total_weight,
                "high": high / total_weight,
                "low": low / total_weight,
            }
        )
    return combined


def _resolve_holding_history(
    holding: Dict[str, Any],
    histories: Dict[str, Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    code = holding.get("code", "")
    mapping = PORTFOLIO_PROXY_MAP.get(code)
    if not mapping:
        return None

    if "components" in mapping:
        series = _combine_component_histories(mapping["components"], histories)
        if not series:
            return None
        proxy_label = mapping.get("label") or code
        proxy_symbol = "/".join(
            component.get("symbol", "")
            for component in mapping["components"]
            if component.get("symbol")
        )
    else:
        symbol = mapping.get("symbol")
        history = histories.get(symbol)
        if not history:
            return None
        series = history["series"]
        proxy_label = mapping.get("label") or history.get("label") or symbol
        proxy_symbol = symbol or ""

    return {
        "code": code,
        "name": holding.get("name") or code,
        "weight_pct": float(holding.get("weight_pct", 0.0)),
        "proxy_label": proxy_label,
        "proxy_symbol": proxy_symbol,
        "series": series,
    }


def _average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _format_pct(value: float) -> str:
    return f"{value:+.2f}%"


def _calc_period_change(closes: List[float], periods: int) -> float:
    if len(closes) <= periods:
        return 0.0
    base = closes[-periods - 1]
    if not base:
        return 0.0
    return ((closes[-1] - base) / base) * 100.0


def _classify_trend(closes: List[float], highs: List[float], lows: List[float]) -> str:
    close = closes[-1]
    ma20 = _average(closes[-20:])
    ma60 = _average(closes[-60:])
    recent_high = max(highs[-20:])
    previous_high = max(highs[-40:-20])
    recent_low = min(lows[-20:])
    previous_low = min(lows[-40:-20])

    if close > ma20 > ma60 and recent_high >= previous_high and recent_low >= previous_low:
        if close >= recent_high * 0.985:
            return "道氏上升趋势延续"
        return "道氏上升趋势"
    if close < ma20 < ma60 and recent_high <= previous_high and recent_low <= previous_low:
        if close <= recent_low * 1.015:
            return "道氏下降趋势延续"
        return "道氏下降趋势"
    if close >= ma20 and recent_low > previous_low:
        return "趋势转强"
    if close <= ma20 and recent_high < previous_high:
        return "趋势转弱"
    return "震荡整理"


def _classify_short_trend_from_points(points: List[Dict[str, Any]]) -> str:
    closes = [point["close"] for point in points]
    highs = [point["high"] for point in points]
    lows = [point["low"] for point in points]

    if len(closes) < 20:
        return "震荡"

    close = closes[-1]
    ma5 = _average(closes[-5:])
    ma20 = _average(closes[-20:])
    recent_high = max(highs[-5:])
    previous_high = max(highs[-10:-5]) if len(highs) >= 10 else recent_high
    recent_low = min(lows[-5:])
    previous_low = min(lows[-10:-5]) if len(lows) >= 10 else recent_low

    if close >= ma5 >= ma20 and recent_high >= previous_high and recent_low >= previous_low:
        return "多头"
    if close <= ma5 <= ma20 and recent_high <= previous_high and recent_low <= previous_low:
        return "空头"
    return "震荡"


def _classify_long_trend_from_points(points: List[Dict[str, Any]]) -> str:
    closes = [point["close"] for point in points]
    highs = [point["high"] for point in points]
    lows = [point["low"] for point in points]

    if len(closes) < 60:
        return "震荡"

    trend = _classify_trend(closes, highs, lows)
    trend_score = TREND_SCORES.get(trend, 0)
    if trend_score > 0:
        return "多头"
    if trend_score < 0:
        return "空头"
    return "震荡"


def _find_latest_state_turn_date(
    series: List[Dict[str, Any]],
    classifier,
    min_points: int,
) -> str:
    if len(series) < min_points:
        return ""

    states = []
    for end in range(min_points, len(series) + 1):
        window = series[:end]
        state = classifier(window)
        date_text = str(window[-1].get("date", "")).strip()
        if not state or not date_text:
            continue
        states.append((date_text, state))

    if not states:
        return ""

    final_state = states[-1][1]
    turn_date = states[-1][0]
    for date_text, state in reversed(states):
        if state != final_state:
            break
        turn_date = date_text
    return turn_date


def _classify_composite_trend(short_trend: str, long_trend: str) -> str:
    if short_trend == "多头" and long_trend == "多头":
        return "确立多头"
    if short_trend == "空头" and long_trend == "空头":
        return "确立空头"
    if short_trend == "多头" and long_trend != "空头":
        return "转强中"
    if short_trend == "空头" and long_trend != "多头":
        return "转弱中"
    return "震荡中"


def _resolve_composite_turn_date(
    composite_trend: str,
    short_turn_date: str,
    long_turn_date: str,
) -> str:
    if composite_trend in {"确立多头", "确立空头"}:
        return long_turn_date or short_turn_date
    return short_turn_date or long_turn_date


def _describe_ma_state(close: float, ma20: float, ma60: float) -> str:
    if close >= ma20 >= ma60:
        return "价格位于20日线和60日线上方"
    if close <= ma20 <= ma60:
        return "价格位于20日线和60日线下方"
    if close >= ma20 and close <= ma60:
        return "价格回到20日线上方但仍受60日线压制"
    if close <= ma20 and close >= ma60:
        return "价格跌回20日线下方，进入强弱再确认阶段"
    return "均线关系交错，结构仍在选择方向"


def _classify_chan_signal(trend: str, closes: List[float], highs: List[float], lows: List[float]) -> tuple[str, str]:
    close = closes[-1]
    ma20 = _average(closes[-20:])
    previous_20_high = max(highs[-40:-20])
    previous_20_low = min(lows[-40:-20])
    recent_10_high = max(highs[-10:])
    recent_10_low = min(lows[-10:])

    if trend in {"道氏上升趋势延续", "道氏上升趋势"} and close >= previous_20_high * 0.995:
        return "三买突破区", "接近前高突破，若站稳前高可视作主升延续，但追高性价比一般"
    if trend in {"道氏上升趋势延续", "道氏上升趋势", "趋势转强"} and abs(close - ma20) / ma20 <= 0.025:
        return "二买观察区", "回踩20日线不破时，仍可按顺势回撤后的二买观察处理"
    if trend in {"道氏下降趋势延续", "道氏下降趋势"} and close <= previous_20_low * 1.02:
        return "一买仅观察区", "接近前低但尚未确认止跌，只能做左侧观察而不能提前确认反转"
    if trend in {"道氏下降趋势延续", "道氏下降趋势", "趋势转弱"}:
        return "二卖/三卖风险区", "反抽若始终站不回20日线，仍应按弱势修复和减仓风控看待"
    if recent_10_high > previous_20_high and recent_10_low > previous_20_low:
        return "三买候选区", "短线高低点同时抬升，若回踩后继续收在20日线上方，可继续看强"
    return "中枢震荡区", "当前更像震荡整理，等待向上突破前高或向下跌破前低再确认方向"


def _analyze_position(position: Dict[str, Any]) -> Dict[str, Any]:
    series = position["series"]
    closes = [point["close"] for point in series]
    highs = [point["high"] for point in series]
    lows = [point["low"] for point in series]
    close = closes[-1]
    ma20 = _average(closes[-20:])
    ma60 = _average(closes[-60:])
    trend = _classify_trend(closes, highs, lows)
    short_trend = _classify_short_trend_from_points(series)
    long_trend = _classify_long_trend_from_points(series)
    short_turn_date = _find_latest_state_turn_date(series, _classify_short_trend_from_points, 20)
    long_turn_date = _find_latest_state_turn_date(series, _classify_long_trend_from_points, 60)
    composite_trend = _classify_composite_trend(short_trend, long_trend)
    composite_turn_date = _resolve_composite_turn_date(
        composite_trend,
        short_turn_date,
        long_turn_date,
    )
    ma_state = _describe_ma_state(close, ma20, ma60)
    signal_label, signal_reason = _classify_chan_signal(trend, closes, highs, lows)
    deviation_pct = ((close - ma60) / ma60 * 100.0) if ma60 else 0.0
    daily_change_pct = _calc_period_change(closes, 1)
    change_5d_pct = _calc_period_change(closes, 5)
    change_20d_pct = _calc_period_change(closes, 20)

    return {
        "code": position["code"],
        "name": position["name"],
        "weight_pct": position["weight_pct"],
        "proxy_symbol": position.get("proxy_symbol", ""),
        "trend": trend,
        "trend_score": TREND_SCORES.get(trend, 0),
        "short_trend": short_trend,
        "long_trend": long_trend,
        "short_turn_date": short_turn_date,
        "long_turn_date": long_turn_date,
        "composite_trend": composite_trend,
        "composite_turn_date": composite_turn_date,
        "composite_score": COMPOSITE_TREND_SCORES.get(composite_trend, 0),
        "deviation_pct": deviation_pct,
        "daily_change_pct": daily_change_pct,
        "change_5d_pct": change_5d_pct,
        "change_20d_pct": change_20d_pct,
        "signal_label": signal_label,
        "signal_reason": signal_reason,
        "ma_state": ma_state,
        "summary": (
            f"{position['name']}：{trend}，{ma_state}；{signal_label}，{signal_reason}。"
        ),
    }


def _select_focus_positions(analyzed_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen = set()

    def add_position(position: Optional[Dict[str, Any]]) -> None:
        if not position:
            return
        key = position["code"]
        if key in seen:
            return
        seen.add(key)
        selected.append(position)

    by_weight = sorted(analyzed_positions, key=lambda item: item["weight_pct"], reverse=True)
    for position in by_weight[:3]:
        add_position(position)

    strongest = max(
        analyzed_positions,
        key=lambda item: (item["trend_score"], item["weight_pct"]),
        default=None,
    )
    weakest = min(
        analyzed_positions,
        key=lambda item: (item["trend_score"], -item["weight_pct"]),
        default=None,
    )
    add_position(strongest)
    add_position(weakest)

    by_structure = sorted(
        analyzed_positions,
        key=lambda item: (abs(item["trend_score"]), item["weight_pct"]),
        reverse=True,
    )
    for position in by_structure:
        if len(selected) >= 5:
            break
        add_position(position)

    return selected[:5]


def _build_technical_summary(analyzed_positions: List[Dict[str, Any]]) -> str:
    strong = [item["name"] for item in analyzed_positions if item["trend_score"] > 0][:2]
    weak = [item["name"] for item in analyzed_positions if item["trend_score"] < 0][:2]
    neutral = [item["name"] for item in analyzed_positions if item["trend_score"] == 0][:1]

    parts = []
    if strong:
        parts.append(f"日线弹性主要集中在{'、'.join(strong)}")
    if weak:
        parts.append(f"{'、'.join(weak)}偏弱，更多承担防守或等待修复")
    if neutral:
        parts.append(f"{'、'.join(neutral)}仍处于震荡选择方向阶段")
    if not parts:
        return "技术快照覆盖有限，当前更适合做定性观察而非结构判断。"
    return "；".join(parts) + "。"


def _build_holding_view_summary(analyzed_positions: List[Dict[str, Any]]) -> str:
    strong = [item["name"] for item in analyzed_positions if item["trend_score"] > 0][:2]
    weak = [item["name"] for item in analyzed_positions if item["trend_score"] < 0][:2]

    parts = []
    if strong:
        parts.append(f"{ '、'.join(strong) }承担主要技术弹性")
    if weak:
        parts.append(f"{ '、'.join(weak) }更偏防守或弱修复")
    if not parts:
        parts.append("大部分仓位仍在震荡区间，组合更适合控制节奏而非追价")
    return "从组合暴露看，" + "；".join(parts) + "。"


def _build_trend_table(analyzed_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        analyzed_positions,
        key=lambda item: (
            item.get("composite_score", 0),
            item.get("deviation_pct", 0.0),
            item.get("change_20d_pct", 0.0),
            item.get("weight_pct", 0.0),
        ),
        reverse=True,
    )

    rows = []
    for index, item in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": index,
                "name": item["name"],
                "code": item["code"],
                "proxy_symbol": item.get("proxy_symbol", ""),
                "deviation_pct": round(item.get("deviation_pct", 0.0), 2),
                "composite_trend": item.get("composite_trend", "震荡中"),
                "composite_turn_date": item.get("composite_turn_date", ""),
                "daily_change_pct": round(item.get("daily_change_pct", 0.0), 2),
                "change_5d_pct": round(item.get("change_5d_pct", 0.0), 2),
                "change_20d_pct": round(item.get("change_20d_pct", 0.0), 2),
                "short_trend": item.get("short_trend", "震荡"),
                "short_turn_date": item.get("short_turn_date", ""),
                "long_trend": item.get("long_trend", "震荡"),
                "long_turn_date": item.get("long_turn_date", ""),
            }
        )
    return rows


def _render_trend_table_prompt(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []

    lines = [
        "- 持仓趋势表：",
        "  排序 | 名称 | 偏离率 | 综合趋势 | 转变日 | 代码 | 今日 | 5日 | 20日 | 短线 | 长线",
    ]
    for row in rows:
        short_view = row.get("short_trend", "震荡")
        short_date = str(row.get("short_turn_date", "")).strip()
        long_view = row.get("long_trend", "震荡")
        long_date = str(row.get("long_turn_date", "")).strip()
        if short_date:
            short_view = f"{short_view}({short_date})"
        if long_date:
            long_view = f"{long_view}({long_date})"
        lines.append(
            "  {rank} | {name} | {deviation} | {composite} | {turn_date} | {code} | {daily} | {change_5d} | {change_20d} | {short_view} | {long_view}".format(
                rank=row.get("rank", "-"),
                name=row.get("name", "-"),
                deviation=_format_pct(float(row.get("deviation_pct", 0.0))),
                composite=row.get("composite_trend", "震荡中"),
                turn_date=row.get("composite_turn_date", "") or "-",
                code=row.get("code", "-"),
                daily=_format_pct(float(row.get("daily_change_pct", 0.0))),
                change_5d=_format_pct(float(row.get("change_5d_pct", 0.0))),
                change_20d=_format_pct(float(row.get("change_20d_pct", 0.0))),
                short_view=short_view,
                long_view=long_view,
            )
        )
    return lines


def _build_prompt_text(snapshot: Dict[str, Any]) -> str:
    lines = [
        "以下为默认持仓的日线技术面快照，仅用于代理结构判断，不得当作精确交易信号。",
        f"- 技术总览：{snapshot['technical_summary']}",
        f"- 组合视角：{snapshot['holding_view_summary']}",
        "- 重点技术信号：",
    ]
    lines.extend(f"  - {signal}" for signal in snapshot["technical_signals"])
    lines.extend(_render_trend_table_prompt(snapshot.get("trend_table", [])))
    return "\n".join(lines)


def build_portfolio_technical_snapshot(
    holdings: List[Dict[str, Any]],
    timeout: float = 3.0,
) -> Dict[str, Any]:
    if not holdings:
        return {
            "prompt_text": "默认持仓为空，无法生成技术面快照。",
        }

    histories = _fetch_histories(_collect_symbols(holdings), timeout)
    resolved_positions = []
    for holding in holdings:
        resolved = _resolve_holding_history(holding, histories)
        if resolved:
            resolved_positions.append(resolved)

    if not resolved_positions:
        return {
            "prompt_text": "技术面快照暂不可用，请不要编造道氏趋势或缠论买卖点，只做定性观察。",
        }

    analyzed_positions = [_analyze_position(position) for position in resolved_positions]
    focus_positions = _select_focus_positions(analyzed_positions)
    technical_signals = [position["summary"] for position in focus_positions]
    trend_table = _build_trend_table(analyzed_positions)
    snapshot = {
        "technical_summary": _build_technical_summary(analyzed_positions),
        "technical_signals": technical_signals,
        "holding_view_summary": _build_holding_view_summary(analyzed_positions),
        "trend_table": trend_table,
    }
    snapshot["prompt_text"] = _build_prompt_text(snapshot)
    return snapshot