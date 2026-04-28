# coding=utf-8
"""
AI 历史回看模块 (Phase 4 - minimal skeleton).

遍历 `output/html/<date>/` 目录下的 AI 分析产物，抽取 `key_message_impacts`
与 `portfolio_summary` 中的判断，生成最近 N 天的"过去预测 vs 当前状态"摘要
Markdown，便于人工评估。

用法：
    python -m trendradar.ai.retrospective --days 7 --out output/retrospective.md

Deliberate simplicity:
- 仅读取已有 HTML 文件中嵌入的 `data-ai-json="..."` 属性（若存在），或退化为
  通过 `output/latest/` 下 `current.html` 的 AI 区块文本做基础统计。
- 不在 Phase 4 中自动打分；输出结构方便后续接入 A/B 评估。

BEGIN BY wangsikan@kuaishou.com: Phase 4 retrospective scaffold.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RetrospectiveEntry:
    date: str
    html_path: str
    key_message_impacts: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_ratings: List[Dict[str, Any]] = field(default_factory=list)
    report_overview: str = ""
    raw_excerpt: str = ""


def _iter_report_dirs(output_html_dir: str, days: int) -> List[str]:
    """Return latest `days` date sub-directories under output/html/."""
    if not os.path.isdir(output_html_dir):
        return []
    entries: List[str] = []
    for name in os.listdir(output_html_dir):
        full = os.path.join(output_html_dir, name)
        if not os.path.isdir(full):
            continue
        if name == "latest":
            continue
        # accept YYYY-MM-DD
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", name):
            entries.append(name)
    entries.sort(reverse=True)
    return entries[:days]


def _pick_latest_html(date_dir: str) -> Optional[str]:
    """Pick the most recent .html file within a date directory."""
    if not os.path.isdir(date_dir):
        return None
    candidates = [
        os.path.join(date_dir, name)
        for name in os.listdir(date_dir)
        if name.endswith(".html")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


_AI_JSON_ATTR_RE = re.compile(
    r'data-ai-json="([^"]+)"', re.S,
)
_AI_SECTION_RE = re.compile(
    r'<div class="ai-analysis">(.*?)</div>\s*</div>', re.S,
)


def _parse_report_html(html_path: str, date: str) -> RetrospectiveEntry:
    entry = RetrospectiveEntry(date=date, html_path=html_path)
    try:
        with open(html_path, "r", encoding="utf-8") as fp:
            content = fp.read()
    except OSError:
        return entry

    match = _AI_JSON_ATTR_RE.search(content)
    if match:
        try:
            payload = json.loads(match.group(1).replace("&quot;", '"'))
            if isinstance(payload, dict):
                entry.report_overview = str(payload.get("report_overview", ""))[:400]
                key_impacts = payload.get("key_message_impacts", [])
                if isinstance(key_impacts, list):
                    entry.key_message_impacts = [
                        {
                            "id": item.get("id", ""),
                            "industry": item.get("industry", ""),
                            "core_view": item.get("core_view", ""),
                        }
                        for item in key_impacts
                        if isinstance(item, dict)
                    ]
                portfolio = payload.get("portfolio_summary", {})
                if isinstance(portfolio, dict):
                    in_actions = portfolio.get("in_portfolio_actions", [])
                    out_actions = portfolio.get("out_of_portfolio_actions", [])
                    if isinstance(in_actions, list):
                        entry.portfolio_ratings.extend(
                            {
                                "side": "in",
                                "name": item.get("name") or item.get("code", ""),
                                "rating": item.get("rating", ""),
                            }
                            for item in in_actions
                            if isinstance(item, dict)
                        )
                    if isinstance(out_actions, list):
                        entry.portfolio_ratings.extend(
                            {
                                "side": "out",
                                "name": item.get("target", ""),
                                "rating": item.get("rating", ""),
                            }
                            for item in out_actions
                            if isinstance(item, dict)
                        )
        except (ValueError, TypeError):
            pass

    if not entry.report_overview:
        section = _AI_SECTION_RE.search(content)
        if section:
            excerpt = re.sub(r"<[^>]+>", " ", section.group(1))
            excerpt = re.sub(r"\s+", " ", excerpt).strip()
            entry.raw_excerpt = excerpt[:600]

    return entry


def build_retrospective(
    output_html_dir: str = os.path.join("output", "html"),
    days: int = 7,
) -> List[RetrospectiveEntry]:
    entries: List[RetrospectiveEntry] = []
    for date in _iter_report_dirs(output_html_dir, days):
        html = _pick_latest_html(os.path.join(output_html_dir, date))
        if not html:
            continue
        entries.append(_parse_report_html(html, date))
    return entries


def render_markdown(entries: List[RetrospectiveEntry]) -> str:
    """Render a minimal markdown for human review."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = [
        f"# AI 历史回看 (生成于 {now})",
        "",
        f"共 {len(entries)} 天样本。请结合当前行情人工核对每条判断是否兑现。",
        "",
    ]
    if not entries:
        lines.append("_未在 `output/html/<date>/` 下找到可解析的报告。_")
        return "\n".join(lines)

    for entry in entries:
        lines.append(f"## {entry.date}")
        lines.append(f"- 报告文件：`{entry.html_path}`")
        if entry.report_overview:
            lines.append(f"- 核心判断（摘录）：{entry.report_overview}")
        elif entry.raw_excerpt:
            lines.append(f"- 摘录（未解析到结构化 JSON）：{entry.raw_excerpt}")
        if entry.key_message_impacts:
            lines.append("- 重点消息：")
            for item in entry.key_message_impacts[:5]:
                lines.append(
                    f"  - [{item.get('id', '')}] "
                    f"{item.get('industry', '')} — {item.get('core_view', '')}"
                )
        if entry.portfolio_ratings:
            lines.append("- 评级：")
            for rating in entry.portfolio_ratings[:10]:
                lines.append(
                    f"  - [{rating.get('side', '')}] "
                    f"{rating.get('name', '')} → {rating.get('rating', '')}"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="AI analysis retrospective")
    parser.add_argument("--days", type=int, default=7, help="look-back window in days")
    parser.add_argument(
        "--html-dir",
        default=os.path.join("output", "html"),
        help="root directory of per-day html reports",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("output", "retrospective.md"),
        help="output markdown file path",
    )
    args = parser.parse_args()

    entries = build_retrospective(args.html_dir, args.days)
    markdown = render_markdown(entries)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        fp.write(markdown)
    print(f"[retrospective] 已生成 {args.out}（{len(entries)} 天样本）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# END BY wangsikan@kuaishou.com
