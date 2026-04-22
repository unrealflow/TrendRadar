# coding=utf-8
"""
AI 分析器模块

调用 AI 大模型对热点新闻进行深度分析
基于 LiteLLM 统一接口，支持 100+ AI 提供商
"""

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional

from trendradar.ai.client import AIClient
from trendradar.ai.portfolio_market import build_portfolio_market_snapshot
from trendradar.ai.prompt_loader import load_prompt_template
from trendradar.ai.technical_analysis import build_portfolio_technical_snapshot


@dataclass
class AIAnalysisResult:
    """AI 分析结果"""
    # 研报式结构 (v3.3.0)
    report_overview: str = ""             # 研报摘要 / 核心判断
    key_message_impacts: List[Dict[str, Any]] = field(default_factory=list)  # 仅展示高影响消息及传导链

    # 兼容旧结构 (v3.2.0)
    message_impacts: List[Dict[str, Any]] = field(default_factory=list)  # 旧版按消息编号输出的行业映射
    personal_layer: str = ""             # 个人层 — 个人投资者行为与情绪
    regional_layer: str = ""             # 地区层 — 区域市场联动与机会
    social_layer: str = ""               # 社会层 — 行业趋势与社会热点
    national_layer: str = ""              # 国家层 — 宏观政策与国际关系
    tech_layer: str = ""                  # 科技层 — 重大技术进展与影响
    portfolio_summary: Dict[str, Any] = field(default_factory=dict)  # 持仓汇总报告

    # 基础元数据
    raw_response: str = ""               # 原始响应
    success: bool = False                # 是否成功
    skipped: bool = False                # 是否因无内容跳过（非失败）
    error: str = ""                      # 错误信息

    # 新闻数量统计
    total_news: int = 0                  # 总新闻数（热榜+RSS）
    analyzed_news: int = 0               # 实际分析的新闻数
    max_news_limit: int = 0              # 分析上限配置值
    hotlist_count: int = 0               # 热榜新闻数
    rss_count: int = 0                   # RSS 新闻数
    ai_mode: str = ""                    # AI 分析使用的模式 (daily/current/incremental)


class AIAnalyzer:
    """AI 分析器"""

    IN_PORTFOLIO_RATINGS = {"加仓", "减仓", "卖出"}
    OUT_OF_PORTFOLIO_RATINGS = {"买入", "观望"}
    CHAIN_TARGET_SUFFIXES = (
        "需求持续走强",
        "需求明显走强",
        "需求快速走强",
        "需求走强",
        "需求提升",
        "需求增加",
        "需求扩张",
        "需求改善",
        "需求上行",
        "订单兑现",
        "订单改善",
        "订单增加",
        "订单放量",
        "价格上涨",
        "价格上行",
        "价格抬升",
        "价格回落",
        "价格下行",
        "预期上修",
        "预期提升",
        "预期改善",
        "逻辑强化",
        "逻辑改善",
        "商业化提速",
        "商业化改善",
        "商业化落地",
        "效率提升",
        "效率改善",
        "受益",
        "承压",
        "走强",
        "改善",
        "提升",
        "加速",
        "扩容",
        "上涨",
        "下跌",
        "回落",
        "收缩",
        "紧张",
        "强化",
        "提速",
        "增加",
        "上修",
        "升温",
        "抬升",
    )
    GENERIC_TARGET_LABELS = {
        "科技",
        "ai",
        "人工智能",
        "算力",
        "半导体",
        "机器人",
        "数据中心",
        "应用层",
        "云厂商",
        "功耗密度",
        "通胀预期",
        "市场风险偏好",
        "成长制造",
        "机房效率",
        "推理成本",
        "商业化",
        "资本开支",
        "ai集群",
        "ai集群建设",
        "ai集群扩容",
    }
    DEFAULT_STAGED_SECTIONS = [
        "report_overview",
        "key_message_impacts",
        "portfolio_summary",
    ]
    VALID_STAGED_SECTIONS = set(DEFAULT_STAGED_SECTIONS)
    MARKET_SUMMARY_KEYWORDS = {
        "科技",
        "港股",
        "宽松",
        "预期",
        "反弹",
        "分化",
        "地缘",
        "风险偏好",
        "高股息",
        "成长",
        "避险",
        "降息",
        "油气",
        "黄金",
    }

    def __init__(
        self,
        ai_config: Dict[str, Any],
        analysis_config: Dict[str, Any],
        get_time_func: Callable,
        debug: bool = False,
    ):
        """
        初始化 AI 分析器

        Args:
            ai_config: AI 模型配置（LiteLLM 格式）
            analysis_config: AI 分析功能配置（language, prompt_file 等）
            get_time_func: 获取当前时间的函数
            debug: 是否开启调试模式
        """
        self.ai_config = ai_config
        self.analysis_config = analysis_config
        self.get_time_func = get_time_func
        self.debug = debug

        # 创建 AI 客户端（基于 LiteLLM）
        self.client = AIClient(ai_config)

        # 验证配置
        valid, error = self.client.validate_config()
        if not valid:
            print(f"[AI] 配置警告: {error}")

        # 从分析配置获取功能参数
        self.max_news = analysis_config.get("MAX_NEWS_FOR_ANALYSIS", 50)
        self.include_rss = analysis_config.get("INCLUDE_RSS", True)
        self.include_rank_timeline = analysis_config.get("INCLUDE_RANK_TIMELINE", False)
        self.include_standalone = analysis_config.get("INCLUDE_STANDALONE", False)
        self.enable_portfolio_market_snapshot = analysis_config.get(
            "ENABLE_PORTFOLIO_MARKET_SNAPSHOT",
            True,
        )
        self.enable_portfolio_technical_snapshot = analysis_config.get(
            "ENABLE_PORTFOLIO_TECHNICAL_SNAPSHOT",
            False,
        )
        self.language = analysis_config.get("LANGUAGE", "Chinese")
        self.staged_mode = analysis_config.get("STAGED_MODE", False)
        self.mcp_runtime_config = None
        if analysis_config.get("ENABLE_MCP", False):
            self.mcp_runtime_config = {
                "ENABLED": True,
                "API_KEY": self.client.api_key or self.ai_config.get("API_KEY", ""),
                "API_HOST": analysis_config.get("MCP_API_HOST", ""),
                "CONFIG_FILE": analysis_config.get("MCP_CONFIG_FILE", ""),
                "SERVERS": analysis_config.get("MCP_SERVERS", ["MiniMax"]),
                "MAX_TOOL_ROUNDS": analysis_config.get("MCP_MAX_TOOL_ROUNDS", 4),
            }
        configured_staged_sections = analysis_config.get(
            "STAGED_SECTIONS",
            self.DEFAULT_STAGED_SECTIONS,
        )
        if not isinstance(configured_staged_sections, list):
            configured_staged_sections = self.DEFAULT_STAGED_SECTIONS
        self.staged_sections = [
            section
            for section in configured_staged_sections
            if section in self.VALID_STAGED_SECTIONS
        ] or list(self.DEFAULT_STAGED_SECTIONS)

        # 加载提示词模板
        self.system_prompt, self.user_prompt_template = load_prompt_template(
            analysis_config.get("PROMPT_FILE", "ai_analysis_prompt.txt"),
            label="AI",
        )
        self.portfolio_holdings = self._extract_portfolio_holding_rows(
            self.user_prompt_template,
        )
        self.portfolio_holding_codes, self.portfolio_holding_names = self._extract_portfolio_holdings(
            self.user_prompt_template,
        )

    def analyze(
        self,
        stats: List[Dict],
        rss_stats: Optional[List[Dict]] = None,
        report_mode: str = "daily",
        report_type: str = "当日汇总",
        platforms: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        standalone_data: Optional[Dict] = None,
    ) -> AIAnalysisResult:
        """
        执行 AI 分析

        Args:
            stats: 热榜统计数据
            rss_stats: RSS 统计数据
            report_mode: 报告模式
            report_type: 报告类型
            platforms: 平台列表
            keywords: 关键词列表

        Returns:
            AIAnalysisResult: 分析结果
        """
        
        # 打印配置信息方便调试
        model = self.ai_config.get("MODEL", "unknown")
        api_key = self.client.api_key or ""
        api_base = self.ai_config.get("API_BASE", "")
        masked_key = f"{api_key[:5]}******" if len(api_key) >= 5 else "******"
        model_display = model.replace("/", "/\u200b") if model else "unknown"

        print(f"[AI] 模型: {model_display}")
        print(f"[AI] Key : {masked_key}")

        if api_base:
            print(f"[AI] 接口: 存在自定义 API 端点")

        timeout = self.ai_config.get("TIMEOUT", 120)
        max_tokens = self.ai_config.get("MAX_TOKENS", 5000)
        print(f"[AI] 参数: timeout={timeout}, max_tokens={max_tokens}")

        if not self.client.api_key:
            return AIAnalysisResult(
                success=False,
                error="未配置 AI API Key，请在 config.yaml 或环境变量 AI_API_KEY 中设置"
            )

        # 准备新闻内容并获取统计数据
        news_content, rss_content, hotlist_total, rss_total, analyzed_count, message_catalog = self._prepare_news_content(stats, rss_stats)
        total_news = hotlist_total + rss_total

        if not news_content and not rss_content:
            return AIAnalysisResult(
                success=False,
                skipped=True,
                error="本轮无新增热点内容，跳过 AI 分析",
                total_news=total_news,
                hotlist_count=hotlist_total,
                rss_count=rss_total,
                analyzed_news=0,
                max_news_limit=self.max_news
            )

        # 构建提示词
        current_time = self.get_time_func().strftime("%Y-%m-%d %H:%M:%S")

        # 提取关键词
        if not keywords:
            keywords = [s.get("word", "") for s in stats if s.get("word")] if stats else []

        # 使用安全的字符串替换，避免模板中其他花括号（如 JSON 示例）被误解析
        user_prompt = self.user_prompt_template
        portfolio_market_snapshot = self._build_portfolio_market_snapshot()
        portfolio_market_snapshot_text = self._resolve_portfolio_market_snapshot_text(
            portfolio_market_snapshot,
        )
        portfolio_technical_snapshot = self._build_portfolio_technical_snapshot()
        portfolio_technical_snapshot_text = self._resolve_portfolio_technical_snapshot_text(
            portfolio_technical_snapshot,
        )
        user_prompt = user_prompt.replace("{report_mode}", report_mode)
        user_prompt = user_prompt.replace("{report_type}", report_type)
        user_prompt = user_prompt.replace("{current_time}", current_time)
        user_prompt = user_prompt.replace("{news_count}", str(hotlist_total))
        user_prompt = user_prompt.replace("{rss_count}", str(rss_total))
        user_prompt = user_prompt.replace("{platforms}", ", ".join(platforms) if platforms else "多平台")
        user_prompt = user_prompt.replace("{keywords}", ", ".join(keywords[:20]) if keywords else "无")
        user_prompt = user_prompt.replace("{news_content}", news_content)
        user_prompt = user_prompt.replace("{rss_content}", rss_content)
        user_prompt = user_prompt.replace("{language}", self.language)
        user_prompt = user_prompt.replace(
            "{portfolio_market_snapshot}",
            portfolio_market_snapshot_text,
        )
        user_prompt = user_prompt.replace(
            "{portfolio_technical_snapshot}",
            portfolio_technical_snapshot_text,
        )

        # 构建独立展示区内容
        standalone_content = ""
        if self.include_standalone and standalone_data:
            standalone_content = self._prepare_standalone_content(standalone_data)
        user_prompt = user_prompt.replace("{standalone_content}", standalone_content)

        if self.debug:
            print("\n" + "=" * 80)
            print("[AI 调试] 发送给 AI 的完整提示词")
            print("=" * 80)
            if self.system_prompt:
                print("\n--- System Prompt ---")
                print(self.system_prompt)
            print("\n--- User Prompt ---")
            print(user_prompt)
            print("=" * 80 + "\n")

        # 调用 AI API（使用 LiteLLM）
        try:
            if self.staged_mode:
                print(f"[AI] 启用分板块分析: {', '.join(self.staged_sections)}")
                try:
                    result = self._analyze_in_stages(user_prompt)
                except Exception as stage_error:
                    print(f"[AI] 分板块分析失败，回退单次调用: {stage_error}")
                    result = self._execute_analysis_prompt(user_prompt)
            else:
                result = self._execute_analysis_prompt(user_prompt)

            # 如果配置未启用 RSS 分析，清空 national_layer 中的 RSS 相关内容（如果 AI 返回了的话）
            # 注：v3.0.0 中 RSS 内容已整合到 national_layer，不再有独立的 rss_insights 字段

            # 如果配置未启用 standalone 分析，清空 social_layer 中的独立展示区相关容
            # 注：v3.0.0 中 standalone 已整合到 social_layer，不再有独立的 standalone_summaries 字段

            # 将编号映射回原始消息，便于渲染时展示标题和来源
            selected_impacts = result.key_message_impacts or result.message_impacts
            selected_impacts = self._enrich_message_impacts(
                selected_impacts,
                message_catalog,
            )
            result.key_message_impacts = selected_impacts
            result.message_impacts = selected_impacts
            result.portfolio_summary = self._normalize_portfolio_summary(
                result.portfolio_summary,
                selected_impacts,
                message_catalog,
                portfolio_market_snapshot=portfolio_market_snapshot,
                portfolio_technical_snapshot=portfolio_technical_snapshot,
                report_overview=result.report_overview,
            )

            # 填充统计数据
            result.total_news = total_news
            result.hotlist_count = hotlist_total
            result.rss_count = rss_total
            result.analyzed_news = analyzed_count
            result.max_news_limit = self.max_news
            result.ai_mode = report_mode
            return result
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            # 截断过长的错误消息
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            friendly_msg = f"AI 分析失败 ({error_type}): {error_msg}"

            return AIAnalysisResult(
                success=False,
                error=friendly_msg
            )

    def _execute_analysis_prompt(self, user_prompt: str) -> AIAnalysisResult:
        """执行一次分析提示词调用，并在必要时尝试修复 JSON。"""
        response = self._call_ai(user_prompt)
        result = self._parse_response(response)

        if result.error and "JSON 解析错误" in result.error:
            print(f"[AI] JSON 解析失败，尝试让 AI 修复...")
            retry_result = self._retry_fix_json(response, result.error)
            if retry_result and retry_result.success and not retry_result.error:
                print("[AI] JSON 修复成功")
                retry_result.raw_response = response
                return retry_result
            print("[AI] JSON 修复失败，使用原始文本兜底")

        return result

    def _analyze_in_stages(self, base_user_prompt: str) -> AIAnalysisResult:
        """按板块分阶段调用模型，并聚合结果。"""
        merged_result = AIAnalysisResult(success=True)
        stage_context: Dict[str, Any] = {}
        raw_parts = []

        for index, section in enumerate(self.staged_sections, start=1):
            print(f"[AI] 分板块 {index}/{len(self.staged_sections)}: {section}")
            stage_prompt = self._build_stage_prompt(base_user_prompt, section, stage_context)
            stage_result = self._execute_analysis_prompt(stage_prompt)

            if not stage_result.success:
                raise RuntimeError(stage_result.error or f"{section} 阶段失败")

            raw_parts.append(f"[{section}]\n{stage_result.raw_response}")
            self._merge_stage_result(merged_result, stage_result, section)
            stage_context = self._build_stage_context(merged_result)

        merged_result.raw_response = "\n\n".join(raw_parts)
        return merged_result

    def _build_stage_prompt(
        self,
        base_user_prompt: str,
        section: str,
        stage_context: Dict[str, Any],
    ) -> str:
        """在基础提示词上叠加板块级约束。"""
        stage_instructions = {
            "report_overview": (
                "当前只生成 report_overview。"
                "只返回 JSON 对象 {\"report_overview\": \"...\"}。"
                "不要返回 key_message_impacts、portfolio_summary 或其他字段。"
            ),
            "key_message_impacts": (
                "当前只生成 key_message_impacts。"
                "只返回 JSON 对象 {\"key_message_impacts\": [...]}。"
                "每条必须包含 id、industry、core_view、direct_chain、indirect_chain。"
                "不要返回 report_overview、portfolio_summary 或其他字段。"
            ),
            "portfolio_summary": (
                "当前只生成 portfolio_summary。"
                "只返回 JSON 对象 {\"portfolio_summary\": {...}}。"
                "持仓内只能用 加仓/减仓/卖出；持仓外只能用 买入/观望。"
                "持仓外方向必须来自热点高影响消息或其直接产业链，不得使用持仓表中已有标的。"
                "不要返回 report_overview、key_message_impacts 或其他字段。"
            ),
        }

        prompt_parts = [base_user_prompt.rstrip()]
        if stage_context:
            prompt_parts.extend(
                [
                    "",
                    "### 已完成板块（仅用于保持一致，不要原样复述）",
                    json.dumps(stage_context, ensure_ascii=False, indent=2),
                ]
            )
        prompt_parts.extend(
            [
                "",
                "### 当前分板块任务",
                stage_instructions.get(section, "只返回当前板块对应的 JSON。"),
            ]
        )
        return "\n".join(prompt_parts)

    def _merge_stage_result(
        self,
        merged_result: AIAnalysisResult,
        stage_result: AIAnalysisResult,
        section: str,
    ) -> None:
        """将单板块结果合并到总结果。"""
        if section == "report_overview" and stage_result.report_overview:
            merged_result.report_overview = stage_result.report_overview
        elif section == "key_message_impacts":
            merged_result.key_message_impacts = stage_result.key_message_impacts or stage_result.message_impacts
            merged_result.message_impacts = merged_result.key_message_impacts
        elif section == "portfolio_summary" and stage_result.portfolio_summary:
            merged_result.portfolio_summary = stage_result.portfolio_summary

        if stage_result.error and not merged_result.error:
            merged_result.error = stage_result.error
        if stage_result.skipped:
            merged_result.skipped = True

    def _build_stage_context(self, result: AIAnalysisResult) -> Dict[str, Any]:
        """提取已完成板块，供后续阶段参考。"""
        context: Dict[str, Any] = {}
        if result.report_overview:
            context["report_overview"] = result.report_overview
        if result.key_message_impacts:
            context["key_message_impacts"] = result.key_message_impacts
        if result.portfolio_summary:
            context["portfolio_summary"] = result.portfolio_summary
        return context

    def _prepare_news_content(
        self,
        stats: List[Dict],
        rss_stats: Optional[List[Dict]] = None,
    ) -> tuple:
        """
        准备新闻内容文本（增强版）

        热榜新闻包含：来源、标题、排名范围、时间范围、出现次数
        RSS 包含：来源、标题、发布时间

        Returns:
            tuple: (news_content, rss_content, hotlist_total, rss_total, analyzed_count, message_catalog)
        """
        news_lines = []
        rss_lines = []
        message_catalog = []
        news_count = 0
        rss_count = 0
        hotlist_message_index = 1
        rss_message_index = 1

        # 计算总新闻数
        hotlist_total = sum(len(s.get("titles", [])) for s in stats) if stats else 0
        rss_total = sum(len(s.get("titles", [])) for s in rss_stats) if rss_stats else 0

        # 热榜内容
        if stats:
            for stat in stats:
                word = stat.get("word", "")
                titles = stat.get("titles", [])
                if word and titles:
                    news_lines.append(f"\n关键词: {word} ({len(titles)}条)")
                    for t in titles:
                        if not isinstance(t, dict):
                            continue
                        title = t.get("title", "")
                        if not title:
                            continue

                        # 来源
                        source = t.get("source_name", t.get("source", ""))
                        message_id = f"H{hotlist_message_index:03d}"

                        # 构建行
                        if source:
                            line = f"- {message_id} | [{source}] {title}"
                        else:
                            line = f"- {message_id} | {title}"

                        # 始终显示简化格式：排名范围 + 时间范围 + 出现次数
                        ranks = t.get("ranks", [])
                        if ranks:
                            min_rank = min(ranks)
                            max_rank = max(ranks)
                            rank_str = f"{min_rank}" if min_rank == max_rank else f"{min_rank}-{max_rank}"
                        else:
                            rank_str = "-"

                        first_time = t.get("first_time", "")
                        last_time = t.get("last_time", "")
                        time_str = self._format_time_range(first_time, last_time)

                        appear_count = t.get("count", 1)

                        line += f" | 排名:{rank_str} | 时间:{time_str} | 出现:{appear_count}次"

                        # 开启完整时间线时，额外添加轨迹
                        if self.include_rank_timeline:
                            rank_timeline = t.get("rank_timeline", [])
                            timeline_str = self._format_rank_timeline(rank_timeline)
                            line += f" | 轨迹:{timeline_str}"

                        news_lines.append(line)
                        message_catalog.append(
                            {
                                "id": message_id,
                                "title": title,
                                "source": source,
                                "source_type": "hotlist",
                                "keyword": word,
                            }
                        )

                        hotlist_message_index += 1
                        news_count += 1
                        if news_count >= self.max_news:
                            break
                if news_count >= self.max_news:
                    break

        # RSS 内容（仅在启用时构建）
        if self.include_rss and rss_stats:
            remaining = self.max_news - news_count
            for stat in rss_stats:
                if rss_count >= remaining:
                    break
                word = stat.get("word", "")
                titles = stat.get("titles", [])
                if word and titles:
                    rss_lines.append(f"\n主题: {word} ({len(titles)}条)")
                    for t in titles:
                        if not isinstance(t, dict):
                            continue
                        title = t.get("title", "")
                        if not title:
                            continue

                        # 来源
                        source = t.get("source_name", t.get("feed_name", ""))
                        message_id = f"R{rss_message_index:03d}"

                        # 发布时间
                        time_display = t.get("time_display", "")

                        # 构建行：编号 | [来源] 标题 | 发布时间
                        if source:
                            line = f"- {message_id} | [{source}] {title}"
                        else:
                            line = f"- {message_id} | {title}"
                        if time_display:
                            line += f" | {time_display}"
                        rss_lines.append(line)
                        message_catalog.append(
                            {
                                "id": message_id,
                                "title": title,
                                "source": source,
                                "source_type": "rss",
                                "keyword": word,
                            }
                        )

                        rss_message_index += 1
                        rss_count += 1
                        if rss_count >= remaining:
                            break

        news_content = "\n".join(news_lines) if news_lines else ""
        rss_content = "\n".join(rss_lines) if rss_lines else ""
        total_count = news_count + rss_count

        return news_content, rss_content, hotlist_total, rss_total, total_count, message_catalog

    def _extract_portfolio_holdings(self, prompt_template: str) -> tuple[set[str], set[str]]:
        """从提示词中的内嵌持仓表提取持仓代码与名称。"""
        codes = set()
        names = set()

        if not prompt_template:
            return codes, names

        for raw_line in prompt_template.splitlines():
            line = raw_line.strip()
            if not line.startswith("|"):
                continue

            match = re.match(r"^\|\s*([A-Za-z0-9.\-]+)\s*\|\s*([^|]+?)\s*\|", line)
            if not match:
                continue

            code = match.group(1).strip()
            name = match.group(2).strip()
            if code == "代码" or name == "名称":
                continue

            normalized_code = self._normalize_text(code)
            normalized_name = self._normalize_text(name)
            if normalized_code:
                codes.add(normalized_code)
            if normalized_name:
                names.add(normalized_name)

        return codes, names

    def _extract_portfolio_holding_rows(self, prompt_template: str) -> List[Dict[str, Any]]:
        """从提示词中的内嵌持仓表提取代码、名称和仓位。"""
        holdings = []

        if not prompt_template:
            return holdings

        for raw_line in prompt_template.splitlines():
            line = raw_line.strip()
            if not line.startswith("|"):
                continue

            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) < 3:
                continue

            code, name, weight_text = parts[:3]
            if code == "代码" or name == "名称":
                continue

            weight_match = re.search(r"(\d+(?:\.\d+)?)%", weight_text)
            if not weight_match:
                continue

            holdings.append(
                {
                    "code": code,
                    "name": name,
                    "weight_pct": float(weight_match.group(1)),
                }
            )

        return holdings

    def _normalize_text(self, value: str) -> str:
        """将文本标准化，便于做代码/名称匹配。"""
        if not value:
            return ""
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value.lower())

    def _iter_text_values(self, value: Any) -> List[str]:
        """将候选值统一展开为文本列表。"""
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, (list, tuple, set)):
            texts = []
            for item in value:
                if not isinstance(item, str):
                    continue
                text = item.strip()
                if text:
                    texts.append(text)
            return texts
        return []

    def _extract_chain_segments(self, chain_text: str) -> List[str]:
        """从传导链中拆出可用于匹配的关键节点。"""
        if not isinstance(chain_text, str) or not chain_text.strip():
            return []

        segments = []
        for raw_part in re.split(r"\s*(?:->|→|=>|➡|⟶)\s*|\n+", chain_text):
            part = raw_part.strip(" ：:;；，,。.()（）[]【】")
            if part:
                segments.append(part)
        return segments

    def _clean_chain_target(self, value: str) -> str:
        """去掉链条末端的状态描述，只保留更像标的的名词短语。"""
        text = value.strip(" ：:;；，,。.()（）[]【】")
        for suffix in self.CHAIN_TARGET_SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)].strip(" ：:;；，,。.()（）[]【】")
                break
        return text

    def _looks_like_tradeable_target(self, value: str) -> bool:
        """用轻量启发式过滤明显不是可交易方向的过程描述。"""
        normalized_value = self._normalize_text(value)
        if len(normalized_value) < 2:
            return False
        if normalized_value in {self._normalize_text(item) for item in self.GENERIC_TARGET_LABELS}:
            return False
        if normalized_value.endswith((
            self._normalize_text("需求"),
            self._normalize_text("预期"),
            self._normalize_text("成本"),
            self._normalize_text("价格"),
            self._normalize_text("效率"),
            self._normalize_text("商业化"),
        )):
            return False
        return True

    def _extract_terminal_chain_targets(self, chain_text: str) -> List[str]:
        """从直接传导链的末端提取更像细分标的的方向。"""
        segments = self._extract_chain_segments(chain_text)
        if not segments:
            return []

        terminal_segment = self._clean_chain_target(segments[-1])
        if not terminal_segment:
            return []

        targets = []
        seen = set()
        for raw_target in re.split(r"\s*(?:和|及|与|、|,|，)\s*", terminal_segment):
            target = raw_target.strip(" ：:;；，,。.()（）[]【】")
            normalized_target = self._normalize_text(target)
            if not normalized_target or normalized_target in seen:
                continue
            if not self._looks_like_tradeable_target(target):
                continue
            seen.add(normalized_target)
            targets.append(target)

        return targets

    def _collect_priority_targets(self, item: Dict[str, Any]) -> List[str]:
        """收集单条高影响消息中的优先细分方向。"""
        targets = []
        seen = set()

        def add_target(value: str) -> None:
            normalized_value = self._normalize_text(value)
            if not normalized_value or normalized_value in seen:
                return
            seen.add(normalized_value)
            targets.append(value.strip())

        for field in ("related_targets", "trade_expression"):
            for value in self._iter_text_values(item.get(field)):
                add_target(value)

        for value in self._extract_terminal_chain_targets(item.get("direct_chain", "")):
            add_target(value)

        industry = item.get("industry", "")
        if (
            isinstance(industry, str)
            and industry.strip()
            and self._looks_like_tradeable_target(industry)
        ):
            add_target(industry)

        return targets

    def _is_portfolio_holding_target(self, target: str) -> bool:
        """判断目标是否属于当前持仓。"""
        normalized_target = self._normalize_text(target)
        if not normalized_target:
            return False
        return (
            normalized_target in self.portfolio_holding_codes
            or normalized_target in self.portfolio_holding_names
        )

    def _collect_hot_direction_candidates(
        self,
        key_message_impacts: List[Dict[str, Any]],
        message_catalog: List[Dict[str, Any]],
    ) -> List[str]:
        """收集可用于持仓外方向校验的热点候选文本。"""
        candidates = []
        seen = set()

        def add_candidate(value: str) -> None:
            normalized_value = self._normalize_text(value)
            if not normalized_value or normalized_value in seen:
                return
            seen.add(normalized_value)
            candidates.append(value.strip())

        for item in key_message_impacts:
            if not isinstance(item, dict):
                continue
            add_candidate(item.get("industry", ""))
            add_candidate(item.get("title", ""))
            add_candidate(item.get("keyword", ""))
            for value in self._iter_text_values(item.get("related_targets")):
                add_candidate(value)
            add_candidate(item.get("trade_expression", ""))
            for value in self._extract_terminal_chain_targets(item.get("direct_chain", "")):
                add_candidate(value)

        for item in message_catalog:
            if not isinstance(item, dict):
                continue
            add_candidate(item.get("keyword", ""))
            add_candidate(item.get("title", ""))

        return candidates

    def _is_hot_derived_target(self, target: str, candidates: List[str]) -> bool:
        """判断持仓外方向是否可从热点消息文本中导出。"""
        normalized_target = self._normalize_text(target)
        if not normalized_target:
            return False

        for candidate in candidates:
            normalized_candidate = self._normalize_text(candidate)
            if not normalized_candidate:
                continue
            if (
                normalized_candidate in normalized_target
                or normalized_target in normalized_candidate
            ):
                return True
        return False

    def _build_hot_watch_fallback(
        self,
        key_message_impacts: List[Dict[str, Any]],
        candidates: List[str],
    ) -> List[Dict[str, Any]]:
        """当模型未给出合规的持仓外方向时，从热点高影响消息回退生成观望项。"""
        fallback_items = []
        seen_targets = set()

        for item in key_message_impacts:
            if not isinstance(item, dict):
                continue

            for target in self._collect_priority_targets(item):
                normalized_target = self._normalize_text(target)
                if not normalized_target or normalized_target in seen_targets:
                    continue
                if self._is_portfolio_holding_target(target):
                    continue
                if not self._is_hot_derived_target(target, candidates):
                    continue

                seen_targets.add(normalized_target)
                fallback_items.append(
                    {
                        "target": target,
                        "rating": "观望",
                        "reason": "由热点消息提炼的持仓外细分方向，当前先保留观察评级，等待进一步催化确认。",
                    }
                )
                if len(fallback_items) >= 3:
                    return fallback_items
                break

        return fallback_items

    def _build_portfolio_market_snapshot(self) -> Dict[str, Any]:
        """Build a lightweight market snapshot for the embedded holdings table."""
        if not self.enable_portfolio_market_snapshot:
            return {
                "prompt_text": "代理行情快照已关闭，请不要编造当日收益数字，只做定性分析。",
            }
        try:
            return build_portfolio_market_snapshot(self.portfolio_holdings)
        except Exception:
            return {
                "prompt_text": "代理行情快照暂不可用，请不要编造当日收益数字，只做定性分析。",
            }

    def _build_portfolio_technical_snapshot(self) -> Dict[str, Any]:
        """Build a lightweight technical snapshot for the embedded holdings table."""
        if not self.enable_portfolio_technical_snapshot:
            return {
                "prompt_text": "技术面快照已关闭，请不要编造道氏趋势或缠论买卖点，只做定性观察。",
            }
        try:
            return build_portfolio_technical_snapshot(self.portfolio_holdings)
        except Exception:
            return {
                "prompt_text": "技术面快照暂不可用，请不要编造道氏趋势或缠论买卖点，只做定性观察。",
            }

    def _resolve_portfolio_market_snapshot_text(self, snapshot: Any) -> str:
        """Resolve prompt text from a snapshot payload or test double."""
        if isinstance(snapshot, str):
            return snapshot
        if isinstance(snapshot, dict):
            prompt_text = snapshot.get("prompt_text", "")
            if isinstance(prompt_text, str) and prompt_text.strip():
                return prompt_text.strip()
        return "代理行情快照暂不可用，请不要编造当日收益数字，只做定性分析。"

    def _resolve_portfolio_technical_snapshot_text(self, snapshot: Any) -> str:
        """Resolve prompt text from a technical snapshot payload or test double."""
        if isinstance(snapshot, str):
            return snapshot
        if isinstance(snapshot, dict):
            prompt_text = snapshot.get("prompt_text", "")
            if isinstance(prompt_text, str) and prompt_text.strip():
                return prompt_text.strip()
        return "技术面快照暂不可用，请不要编造道氏趋势或缠论买卖点，只做定性观察。"

    def _merge_portfolio_market_snapshot(
        self,
        portfolio_summary: Dict[str, Any],
        portfolio_market_snapshot: Any,
    ) -> Dict[str, Any]:
        """Merge factual portfolio market context into the final summary."""
        if not isinstance(portfolio_summary, dict):
            return portfolio_summary
        if not isinstance(portfolio_market_snapshot, dict):
            return portfolio_summary

        merged_summary = dict(portfolio_summary)
        daily_performance = portfolio_market_snapshot.get("daily_performance")
        if isinstance(daily_performance, dict) and daily_performance:
            merged_summary["daily_performance"] = daily_performance

        if not merged_summary.get("market_summary"):
            market_summary = portfolio_market_snapshot.get("market_summary")
            if isinstance(market_summary, str) and market_summary.strip():
                merged_summary["market_summary"] = market_summary.strip()

        if not merged_summary.get("holding_notes"):
            holding_notes = portfolio_market_snapshot.get("holding_notes")
            if isinstance(holding_notes, list) and holding_notes:
                merged_summary["holding_notes"] = holding_notes

        return merged_summary

    def _merge_portfolio_technical_snapshot(
        self,
        portfolio_summary: Dict[str, Any],
        portfolio_technical_snapshot: Any,
    ) -> Dict[str, Any]:
        """Merge technical portfolio context into the final summary."""
        if not isinstance(portfolio_summary, dict):
            return portfolio_summary
        if not isinstance(portfolio_technical_snapshot, dict):
            return portfolio_summary

        merged_summary = dict(portfolio_summary)

        if not merged_summary.get("market_summary"):
            holding_view_summary = portfolio_technical_snapshot.get("holding_view_summary")
            if isinstance(holding_view_summary, str) and holding_view_summary.strip():
                merged_summary["market_summary"] = holding_view_summary.strip()

        if not merged_summary.get("technical_summary"):
            technical_summary = portfolio_technical_snapshot.get("technical_summary")
            if isinstance(technical_summary, str) and technical_summary.strip():
                merged_summary["technical_summary"] = technical_summary.strip()

        if not merged_summary.get("technical_signals"):
            technical_signals = portfolio_technical_snapshot.get("technical_signals")
            if isinstance(technical_signals, list) and technical_signals:
                merged_summary["technical_signals"] = technical_signals

        return merged_summary

    def _extract_labeled_section(self, text: str, label: str) -> str:
        """Extract text from a labeled section like 【盘面小结】."""
        if not isinstance(text, str) or not text.strip():
            return ""

        match = re.search(
            rf"【{re.escape(label)}】[:：]?\s*(.*?)(?=(?:\n\s*\n【)|$)",
            text,
            re.S,
        )
        if not match:
            return ""
        return match.group(1).strip()

    def _extract_summary_clauses(self, text: str) -> List[str]:
        """Split summary text into normalized clauses for lightweight overlap checks."""
        clauses = []
        for raw_part in re.split(r"[，,。；;：:\n]+", text or ""):
            clause = self._normalize_text(raw_part.strip())
            if len(clause) >= 6:
                clauses.append(clause)
        return clauses

    def _should_replace_market_summary(self, report_overview: str, market_summary: str) -> bool:
        """Detect whether portfolio market_summary is repeating the overview narrative."""
        overview_market = self._extract_labeled_section(report_overview, "盘面小结") or report_overview
        normalized_overview = self._normalize_text(overview_market)
        normalized_market = self._normalize_text(market_summary)

        if not normalized_overview or not normalized_market:
            return False
        if normalized_overview in normalized_market or normalized_market in normalized_overview:
            return True

        similarity = SequenceMatcher(None, normalized_overview, normalized_market).ratio()
        shared_clauses = set(self._extract_summary_clauses(overview_market)) & set(
            self._extract_summary_clauses(market_summary)
        )
        shared_keywords = [
            keyword
            for keyword in self.MARKET_SUMMARY_KEYWORDS
            if keyword in overview_market and keyword in market_summary
        ]

        return (
            similarity >= 0.72
            or len(shared_clauses) >= 2
            or (len(shared_clauses) >= 1 and similarity >= 0.45)
            or (similarity >= 0.50 and len(shared_keywords) >= 3)
        )

    def _apply_portfolio_summary_dedupe(
        self,
        portfolio_summary: Dict[str, Any],
        report_overview: str,
        portfolio_market_snapshot: Optional[Any],
        portfolio_technical_snapshot: Optional[Any],
    ) -> Dict[str, Any]:
        """Replace repeated holdings summary with a holding-specific fallback."""
        if not isinstance(portfolio_summary, dict):
            return portfolio_summary

        market_summary = portfolio_summary.get("market_summary")
        if not isinstance(market_summary, str) or not market_summary.strip():
            return portfolio_summary
        if not self._should_replace_market_summary(report_overview, market_summary):
            return portfolio_summary

        replacement = ""
        if isinstance(portfolio_technical_snapshot, dict):
            candidate = portfolio_technical_snapshot.get("holding_view_summary")
            if isinstance(candidate, str) and candidate.strip():
                replacement = candidate.strip()

        if not replacement and isinstance(portfolio_market_snapshot, dict):
            candidate = portfolio_market_snapshot.get("market_summary")
            if isinstance(candidate, str) and candidate.strip():
                replacement = candidate.strip()

        deduped_summary = dict(portfolio_summary)
        if replacement:
            deduped_summary["market_summary"] = replacement
        return deduped_summary

    def _normalize_text_list(self, raw_items: Any) -> List[str]:
        """Keep only non-empty strings while preserving order."""
        if not isinstance(raw_items, list):
            return []

        normalized_items = []
        seen_items = set()
        for item in raw_items:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text or text in seen_items:
                continue
            seen_items.add(text)
            normalized_items.append(text)
        return normalized_items

    def _normalize_matrix_distribution(self, raw_matrix: Any) -> Dict[str, List[str]]:
        """Validate matrix zones so formatter can safely render them."""
        if not isinstance(raw_matrix, dict):
            return {}

        normalized_matrix = {}
        for zone_key in ["黄金区", "需谨慎", "烟蒂区", "双杀区"]:
            items = self._normalize_text_list(raw_matrix.get(zone_key, []))
            if items:
                normalized_matrix[zone_key] = items
        return normalized_matrix

    def _dedupe_portfolio_risk_warnings(
        self,
        portfolio_summary: Dict[str, Any],
        report_overview: str,
    ) -> Dict[str, Any]:
        """Drop portfolio risk warnings that repeat the overview risk section when matrix data exists."""
        if not isinstance(portfolio_summary, dict):
            return portfolio_summary

        matrix_distribution = portfolio_summary.get("matrix_distribution")
        if not isinstance(matrix_distribution, dict) or not any(matrix_distribution.values()):
            return portfolio_summary

        warnings = portfolio_summary.get("risk_warnings")
        if not isinstance(warnings, list) or not warnings:
            return portfolio_summary

        overview_risk = self._extract_labeled_section(report_overview, "风险提醒")
        normalized_overview_risk = self._normalize_text(overview_risk)

        deduped_warnings = []
        seen_warnings = set()
        for warning in warnings:
            if not isinstance(warning, str) or not warning.strip():
                continue

            warning_text = warning.strip()
            normalized_warning = self._normalize_text(warning_text)
            if not normalized_warning or normalized_warning in seen_warnings:
                continue

            if normalized_overview_risk:
                similarity = SequenceMatcher(None, normalized_overview_risk, normalized_warning).ratio()
                if (
                    normalized_warning in normalized_overview_risk
                    or normalized_overview_risk in normalized_warning
                    or similarity >= 0.72
                ):
                    continue

            seen_warnings.add(normalized_warning)
            deduped_warnings.append(warning_text)

        deduped_summary = dict(portfolio_summary)
        deduped_summary["risk_warnings"] = deduped_warnings
        return deduped_summary

    def _normalize_portfolio_summary(
        self,
        portfolio_summary: Dict[str, Any],
        key_message_impacts: List[Dict[str, Any]],
        message_catalog: List[Dict[str, Any]],
        portfolio_market_snapshot: Optional[Any] = None,
        portfolio_technical_snapshot: Optional[Any] = None,
        report_overview: str = "",
    ) -> Dict[str, Any]:
        """校验并修正持仓内外评级，避免模型把持仓内标的写入持仓外。"""
        if not isinstance(portfolio_summary, dict):
            return portfolio_summary

        normalized_summary = dict(portfolio_summary)
        hot_candidates = self._collect_hot_direction_candidates(
            key_message_impacts,
            message_catalog,
        )

        in_actions = []
        for item in normalized_summary.get("in_portfolio_actions", []):
            if not isinstance(item, dict):
                continue

            code = item.get("code", "")
            name = item.get("name", "")
            target_text = code or name
            rating = item.get("rating", "")
            if not self._is_portfolio_holding_target(target_text):
                continue
            if rating not in self.IN_PORTFOLIO_RATINGS:
                continue

            normalized_item = dict(item)
            normalized_item["code"] = code.strip()
            normalized_item["name"] = name.strip()
            in_actions.append(normalized_item)

        out_actions = []
        for item in normalized_summary.get("out_of_portfolio_actions", []):
            if not isinstance(item, dict):
                continue

            target = item.get("target", "")
            rating = item.get("rating", "")
            if not target:
                continue
            if not self._looks_like_tradeable_target(target):
                continue
            if self._is_portfolio_holding_target(target):
                continue
            if not self._is_hot_derived_target(target, hot_candidates):
                continue
            if rating not in self.OUT_OF_PORTFOLIO_RATINGS:
                rating = "观望"

            normalized_item = dict(item)
            normalized_item["target"] = target.strip()
            normalized_item["rating"] = rating
            out_actions.append(normalized_item)

        if not out_actions:
            out_actions = self._build_hot_watch_fallback(
                key_message_impacts,
                hot_candidates,
            )

        normalized_summary["in_portfolio_actions"] = in_actions
        normalized_summary["out_of_portfolio_actions"] = out_actions
        normalized_summary = self._merge_portfolio_market_snapshot(
            normalized_summary,
            portfolio_market_snapshot,
        )
        normalized_summary = self._merge_portfolio_technical_snapshot(
            normalized_summary,
            portfolio_technical_snapshot,
        )
        normalized_summary = self._apply_portfolio_summary_dedupe(
            normalized_summary,
            report_overview,
            portfolio_market_snapshot,
            portfolio_technical_snapshot,
        )
        normalized_summary["matrix_distribution"] = self._normalize_matrix_distribution(
            normalized_summary.get("matrix_distribution")
        )
        normalized_summary["top_opportunities"] = self._normalize_text_list(
            normalized_summary.get("top_opportunities")
        )
        normalized_summary["action_suggestions"] = self._normalize_text_list(
            normalized_summary.get("action_suggestions")
        )
        normalized_summary.pop("risk_warnings", None)
        return normalized_summary

    def _enrich_message_impacts(
        self,
        message_impacts: List[Dict[str, Any]],
        message_catalog: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """将 AI 返回的编号映射回原始消息标题与来源。"""
        if not message_impacts:
            return []

        catalog_by_id = {
            item.get("id", ""): item
            for item in message_catalog
            if item.get("id")
        }
        catalog_order = {
            item.get("id", ""): index
            for index, item in enumerate(message_catalog)
            if item.get("id")
        }

        enriched_impacts = []
        seen_ids = set()
        for item in message_impacts:
            if not isinstance(item, dict):
                continue

            normalized_item = {}
            for key, value in item.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    value = value.strip()
                normalized_item[key] = value

            impact_id = normalized_item.get("id", "")
            if impact_id in seen_ids:
                continue
            if impact_id:
                seen_ids.add(impact_id)

            catalog_item = catalog_by_id.get(impact_id)
            if catalog_item:
                normalized_item.setdefault("title", catalog_item.get("title", ""))
                normalized_item.setdefault("source", catalog_item.get("source", ""))
                normalized_item.setdefault("source_type", catalog_item.get("source_type", ""))
                normalized_item.setdefault("keyword", catalog_item.get("keyword", ""))

            enriched_impacts.append(normalized_item)

        enriched_impacts.sort(
            key=lambda item: catalog_order.get(item.get("id", ""), len(catalog_order))
        )
        return enriched_impacts

    def _call_ai(self, user_prompt: str) -> str:
        """调用 AI API（使用 LiteLLM）"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if self.mcp_runtime_config:
            return self.client.chat(messages, mcp_config=self.mcp_runtime_config)
        return self.client.chat(messages)

    def _retry_fix_json(self, original_response: str, error_msg: str) -> Optional[AIAnalysisResult]:
        """
        JSON 解析失败时，请求 AI 修复 JSON（仅重试一次）

        使用轻量 prompt，不重复原始分析的 system prompt，节省 token。

        Args:
            original_response: AI 原始响应（JSON 格式有误）
            error_msg: JSON 解析的错误信息

        Returns:
            修复后的分析结果，失败时返回 None
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个 JSON 修复助手。用户会提供一段格式有误的 JSON 和错误信息，"
                    "你需要修复 JSON 格式错误并返回正确的 JSON。\n"
                    "常见问题：字符串值内的双引号未转义、缺少逗号、字符串未正确闭合等。\n"
                    "只返回纯 JSON，不要包含 markdown 代码块标记（如 ```json）或任何说明文字。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"以下 JSON 解析失败：\n\n"
                    f"错误：{error_msg}\n\n"
                    f"原始内容：\n{original_response}\n\n"
                    f"请修复以上 JSON 中的格式问题（如值中的双引号改用中文引号「」或转义 \\\"、"
                    f"缺少逗号、不完整的字符串等），保持原始内容语义不变，只修复格式。"
                    f"直接返回修复后的纯 JSON。"
                ),
            },
        ]

        try:
            response = self.client.chat(messages)
            return self._parse_response(response)
        except Exception as e:
            print(f"[AI] 重试修复 JSON 异常: {type(e).__name__}: {e}")
            return None

    def _format_time_range(self, first_time: str, last_time: str) -> str:
        """格式化时间范围（简化显示，只保留时分）"""
        def extract_time(time_str: str) -> str:
            if not time_str:
                return "-"
            # 尝试提取 HH:MM 部分
            if " " in time_str:
                parts = time_str.split(" ")
                if len(parts) >= 2:
                    time_part = parts[1]
                    if ":" in time_part:
                        return time_part[:5]  # HH:MM
            elif ":" in time_str:
                return time_str[:5]
            # 处理 HH-MM 格式
            result = time_str[:5] if len(time_str) >= 5 else time_str
            if len(result) == 5 and result[2] == '-':
                result = result.replace('-', ':')
            return result

        first = extract_time(first_time)
        last = extract_time(last_time)

        if first == last or last == "-":
            return first
        return f"{first}~{last}"

    def _format_rank_timeline(self, rank_timeline: List[Dict]) -> str:
        """格式化排名时间线"""
        if not rank_timeline:
            return "-"

        parts = []
        for item in rank_timeline:
            time_str = item.get("time", "")
            if len(time_str) == 5 and time_str[2] == '-':
                time_str = time_str.replace('-', ':')
            rank = item.get("rank")
            if rank is None:
                parts.append(f"0({time_str})")
            else:
                parts.append(f"{rank}({time_str})")

        return "→".join(parts)

    def _prepare_standalone_content(self, standalone_data: Dict) -> str:
        """
        将独立展示区数据转为文本，注入 AI 分析 prompt

        Args:
            standalone_data: 独立展示区数据 {"platforms": [...], "rss_feeds": [...]}

        Returns:
            格式化的文本内容
        """
        lines = []

        # 热榜平台
        for platform in standalone_data.get("platforms", []):
            platform_id = platform.get("id", "")
            platform_name = platform.get("name", platform_id)
            items = platform.get("items", [])
            if not items:
                continue

            lines.append(f"平台: [{platform_name}]")
            for item in items:
                title = item.get("title", "")
                if not title:
                    continue

                line = f"- {title}"

                # 排名信息
                ranks = item.get("ranks", [])
                if ranks:
                    min_rank = min(ranks)
                    max_rank = max(ranks)
                    rank_str = f"{min_rank}" if min_rank == max_rank else f"{min_rank}-{max_rank}"
                    line += f" | 排名:{rank_str}"

                # 时间范围
                first_time = item.get("first_time", "")
                last_time = item.get("last_time", "")
                if first_time:
                    time_str = self._format_time_range(first_time, last_time)
                    line += f" | 时间:{time_str}"

                # 出现次数
                count = item.get("count", 1)
                if count > 1:
                    line += f" | 出现:{count}次"

                # 排名轨迹（如果启用）
                if self.include_rank_timeline:
                    rank_timeline = item.get("rank_timeline", [])
                    if rank_timeline:
                        timeline_str = self._format_rank_timeline(rank_timeline)
                        line += f" | 轨迹:{timeline_str}"

                lines.append(line)
            lines.append("")

        # RSS 源
        for feed in standalone_data.get("rss_feeds", []):
            feed_id = feed.get("id", "")
            feed_name = feed.get("name", feed_id)
            items = feed.get("items", [])
            if not items:
                continue

            lines.append(f"RSS源: [{feed_name}]")
            for item in items:
                title = item.get("title", "")
                if not title:
                    continue

                line = f"- {title}"
                published_at = item.get("published_at", "")
                if published_at:
                    line += f" | {published_at}"

                lines.append(line)
            lines.append("")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> AIAnalysisResult:
        """解析 AI 响应"""
        result = AIAnalysisResult(raw_response=response)

        if not response or not response.strip():
            result.error = "AI 返回空响应"
            return result

        # 提取 JSON 文本（去掉 markdown 代码块标记）
        json_str = response

        if "```json" in response:
            parts = response.split("```json", 1)
            if len(parts) > 1:
                code_block = parts[1]
                end_idx = code_block.find("```")
                if end_idx != -1:
                    json_str = code_block[:end_idx]
                else:
                    json_str = code_block
        elif "```" in response:
            parts = response.split("```", 2)
            if len(parts) >= 2:
                json_str = parts[1]

        json_str = json_str.strip()
        if not json_str:
            result.error = "提取的 JSON 内容为空"
            result.report_overview = response[:500] + "..." if len(response) > 500 else response
            result.personal_layer = response[:500] + "..." if len(response) > 500 else response
            result.success = True
            return result

        # 第一步：标准 JSON 解析
        data = None
        parse_error = None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            parse_error = e

        # 第二步：json_repair 本地修复
        if data is None:
            try:
                from json_repair import repair_json
                repaired = repair_json(json_str, return_objects=True)
                if isinstance(repaired, dict):
                    data = repaired
                    print("[AI] JSON 本地修复成功（json_repair）")
            except Exception:
                pass

        # 两步都失败，记录错误（后续由 analyze 方法的重试机制处理）
        if data is None:
            if parse_error:
                error_context = json_str[max(0, parse_error.pos - 30):parse_error.pos + 30] if json_str and parse_error.pos else ""
                result.error = f"JSON 解析错误 (位置 {parse_error.pos}): {parse_error.msg}"
                if error_context:
                    result.error += f"，上下文: ...{error_context}..."
            else:
                result.error = "JSON 解析失败"
            # 兜底：使用已提取的 json_str（不含 markdown 标记），避免推送中出现 ```json
            result.report_overview = json_str[:500] + "..." if len(json_str) > 500 else json_str
            result.personal_layer = json_str[:500] + "..." if len(json_str) > 500 else json_str
            result.success = True
            return result

        # 解析成功，提取字段 (v3.3.0 研报结构 + 旧结构兼容)
        try:
            key_message_impacts = data.get("key_message_impacts")
            if isinstance(key_message_impacts, list):
                result.key_message_impacts = [
                    item for item in key_message_impacts if isinstance(item, dict)
                ]
            else:
                message_impacts = data.get("message_impacts", [])
                if isinstance(message_impacts, list):
                    result.key_message_impacts = [
                        item for item in message_impacts if isinstance(item, dict)
                    ]

            result.report_overview = data.get("report_overview", "")

            result.personal_layer = data.get("personal_layer", "")
            result.regional_layer = data.get("regional_layer", "")
            result.social_layer = data.get("social_layer", "")
            result.national_layer = data.get("national_layer", "")
            result.tech_layer = data.get("tech_layer", "")

            # 解析持仓汇总报告（结构化 JSON 对象）
            portfolio = data.get("portfolio_summary", {})
            if isinstance(portfolio, dict):
                result.portfolio_summary = portfolio
            elif isinstance(portfolio, str):
                # 如果 AI 返回的是字符串而非对象，包装为简单结构
                result.portfolio_summary = {"raw_content": portfolio}

            result.success = True
        except (KeyError, TypeError, AttributeError) as e:
            result.error = f"字段提取错误: {type(e).__name__}: {e}"
            result.report_overview = json_str[:500] + "..." if len(json_str) > 500 else json_str
            result.personal_layer = json_str[:500] + "..." if len(json_str) > 500 else json_str
            result.success = True

        return result
