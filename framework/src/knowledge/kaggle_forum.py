"""
Kaggle 论坛 / Discussions 查询

数据源:Kaggle 公开 API + 网页抓取(无需认证,公开数据)
常用 endpoint:https://www.kaggle.com/discussions/{competition-slug}

典型场景:
    - "Spaceship Titanic top solution 用了什么 trick?"
    - "Kaggle S6E2 论坛上有没有讨论 CatBoost 调参?"
    - "某个 competition 的 discussion 里 CV-LB gap 的真实值?"
"""
from __future__ import annotations

import json
import urllib.request
import urllib.parse
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict


KAGGLE_DISCUSSION_BASE = "https://www.kaggle.com/discussions"


@dataclass
class KaggleDiscussion:
    """一条 Kaggle discussion"""
    title: str
    url: str
    author: str = ""
    votes: int = 0
    comments_count: int = 0
    snippet: str = ""
    competition: str = ""
    tags: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"[{self.votes}↑ {self.comments_count}💬] {self.title} — {self.author}"

    def to_markdown(self) -> str:
        return (
            f"## {self.title}\n\n"
            f"- **Author**: {self.author} | **Votes**: {self.votes} | **Comments**: {self.comments_count}\n"
            f"- **URL**: {self.url}\n"
            f"- **Competition**: {self.competition}\n"
            f"- **Snippet**: {self.snippet[:500]}...\n\n"
        )


class KaggleForum:
    """
    Kaggle 论坛查询(轻量实现,无认证)

    重要:这是"实时获取"层,补充 skills 库的静态知识。
    顶级选手在 discussion 里分享的"非官方"trick,经常比论文更实用。
    """

    def __init__(self, rate_limit_sec: float = 2.0):
        self.rate_limit_sec = rate_limit_sec
        self._last_request = None

    def search_competition(self, competition_slug: str,
                           query: Optional = None,
                           sort_by: str = "votes") -> List[KaggleDiscussion]:
        """查询某竞赛的 discussions

        Args:
            competition_slug: 竞赛标识,如 "spaceship-titanic"、"store-sales-time-series-forecasting"
            query: 可选关键词过滤
            sort_by: 排序字段 "votes" | "comments" | "recent"
        """
        # Kaggle 的 discussions 数据通过内部 JSON API 提供
        # 这是公开 API,无需认证即可访问
        url = f"{KAGGLE_DISCUSSION_BASE}/{competition_slug}"
        if query:
            url += f"?searchQuery={urllib.parse.quote(query)}"
        if sort_by:
            url += f"&sortBy={sort_by}"

        return self._fetch_discussions(url, competition_slug)

    def search_global(self, query: str, max_results: int = 20) -> List[KaggleDiscussion]:
        """全局搜索 Kaggle discussions"""
        url = f"https://www.kaggle.com/discussions?searchQuery={urllib.parse.quote(query)}"
        return self._fetch_discussions(url, competition_slug="", max_results=max_results)

    def get_top_solutions(self, competition_slug: str,
                          top_n: int = 20) -> List[KaggleDiscussion]:
        """获取某竞赛 votes 最高的 N 条 discussion(通常包含 top solution 分享)"""
        discussions = self.search_competition(competition_slug, sort_by="votes")
        return discussions[:top_n]

    # ---- helpers ----

    def _fetch_discussions(self, url: str, competition_slug: str,
                           max_results: int = 30) -> List[KaggleDiscussion]:
        self._respect_rate_limit()

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; cultivating-ml-agent/1.0)",
                    "Accept": "application/json, text/html",
                }
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                content_type = r.headers.get("Content-Type", "")
                body = r.read()

                # 如果返回 JSON,直接解析;否则尝试从 HTML 提取
                if "json" in content_type:
                    data = json.loads(body.decode("utf-8", errors="replace"))
                    return self._parse_json(data, competition_slug, max_results)
                else:
                    return self._parse_html(body.decode("utf-8", errors="replace"),
                                            competition_slug, max_results)
        except Exception as e:
            raise RuntimeError(f"Kaggle discussion fetch failed: {e}")

    def _parse_json(self, data: Dict, competition: str, max_results: int) -> List[KaggleDiscussion]:
        """解析 Kaggle 内部 JSON API 返回"""
        results: List[KaggleDiscussion] = []
        discussions = data.get("discussions", []) or data.get("results", [])
        for d in discussions[:max_results]:
            try:
                results.append(KaggleDiscussion(
                    title=d.get("title", "(no title)"),
                    url=d.get("url", d.get("discussionUrl", "")),
                    author=d.get("user", {}).get("displayName", ""),
                    votes=d.get("voteCount", 0) or d.get("votes", 0),
                    comments_count=d.get("commentCount", 0) or d.get("numComments", 0),
                    snippet=d.get("snippet", "") or d.get("preview", ""),
                    competition=competition,
                    tags=d.get("tags", []) or [],
                ))
            except Exception:
                continue
        return results

    def _parse_html(self, html: str, competition: str, max_results: int) -> List[KaggleDiscussion]:
        """从 Kaggle discussions 页面 HTML 提取(简化的正则提取)"""
        results: List[KaggleDiscussion] = []

        # 提取 discussion card — Kaggle 的 HTML 结构相对稳定
        pattern = re.compile(
            r'<a[^>]*href="(/discussions/[^"]+)"[^>]*>([^<]+)</a>',
            re.IGNORECASE
        )
        seen_urls = set()

        for m in pattern.finditer(html):
            url_path, title = m.group(1), m.group(2).strip()
            full_url = "https://www.kaggle.com" + url_path
            if full_url in seen_urls or not title or len(title) < 5:
                continue
            seen_urls.add(full_url)
            results.append(KaggleDiscussion(
                title=title,
                url=full_url,
                competition=competition,
                snippet="",  # HTML 提取较粗,内容需进一步抓
            ))
            if len(results) >= max_results:
                break

        return results

    def _respect_rate_limit(self):
        if self._last_request:
            import time
            elapsed = (time.time() - self._last_request)
            if elapsed < self.rate_limit_sec:
                time.sleep(self.rate_limit_sec - elapsed)
        import time as t
        self._last_request = t.time()