"""
arxiv 论文搜索 — 实时获取最新 ML 论文

API: http://export.arxiv.org/api_query/  (完全免费,无需 API key)
官方文档: https://arxiv.org/help/api

典型用法:
    search = ArxivSearch()
    papers = search.search("time series forecasting", max_results=10)
    for p in papers:
        print(f"{p.title}\\n  {p.authors[0]} et al. ({p.year})\\n  {p.pdf_url}")
"""
from __future__ import annotations

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional


ARXIV_API = "http://export.arxiv.org/api_query/"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass
class ArxivPaper:
    """一篇 arxiv 论文"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str           # ISO date
    updated: str
    pdf_url: str
    abs_url: str
    year: int = 0

    def __repr__(self):
        first_author = self.authors[0] if self.authors else "?"
        return f"[{self.year}] {self.title} — {first_author} et al."

    def to_markdown(self) -> str:
        """渲染成 markdown 片段,可直接追加到 vault"""
        authors_str = ", ".join(self.authors[:5])
        if len(self.authors) > 5:
            authors_str += f" ... +{len(self.authors) - 5}"
        return (
            f"## {self.title}\n\n"
            f"- **Authors**: {authors_str}\n"
            f"- **Year**: {self.year}\n"
            f"- **Categories**: {', '.join(self.categories[:5])}\n"
            f"- **PDF**: {self.pdf_url}\n"
            f"- **Abstract**: {self.abstract}\n\n"
        )


class ArxivSearch:
    """arxiv API 客户端"""

    def __init__(self, rate_limit_sec: float = 3.0, cache_dir: Optional = Optional[str]):
        """
        Args:
            rate_limit_sec: arxiv 建议请求间隔 3 秒
            cache_dir: 可选缓存目录(避免重复请求)
        """
        self.rate_limit_sec = rate_limit_sec
        self.cache_dir = cache_dir
        self._last_request: Optional[datetime] = None

    def search(self, query: str, max_results: int = 10,
               category: Optional = None,
               years_back: Optional = None) -> List[ArxivPaper]:
        """搜索 arxiv 论文

        Args:
            query: 搜索关键词(支持 arxiv 查询语法,如 "ti:transformer AND abs:time series")
            max_results: 最大返回数(arxiv 上限 2000)
            category: 限定类目,如 "cs.LG"(机器学习)、"stat.ML"(统计 ML)
            years_back: 只取最近 N 年
        """
        self._respect_rate_limit()

        q_parts = [query]
        if category:
            q_parts.append(f"cat:{category}")
        if years_back:
            from_date = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y%m%d")
            q_parts.append(f"submittedDate:[{from_date} TO *]")

        params = {
            "search_query": " AND ".join(f"({p})" for p in q_parts),
            "start": 0,
            "max_results": min(max_results, 2000),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        # arxiv API 期望 + 号表示 AND(而非 %20 或 encoded +)
        query_string = urllib.parse.urlencode(params).replace("+", "+")
        url = ARXIV_API + "?" + query_string

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cultivating-ml-agent/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                xml_data = r.read()
        except Exception as e:
            raise RuntimeError(f"arxiv API request failed: {e}")

        return self._parse_xml(xml_data)

    def search_recent(self, category: str = "cs.LG", days: int = 7,
                      max_results: int = 20) -> List[ArxivPaper]:
        """便捷方法:最近 N 天某类目下的论文"""
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        q = f"cat:{category} AND submittedDate:[{from_date} TO *]"
        return self.search(q, max_results=max_results)

    def get_paper(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """根据 arxiv_id 拉单篇论文详情"""
        url = f"{ARXIV_API}?id_list={urllib.parse.quote(arxiv_id)}"
        try:
            self._respect_rate_limit()
            req = urllib.request.Request(url, headers={"User-Agent": "cultivating-ml-agent/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                papers = self._parse_xml(r.read())
            return papers[0] if papers else None
        except Exception:
            return None

    # ---- helpers ----

    def _respect_rate_limit(self):
        if self._last_request:
            elapsed = (datetime.now() - self._last_request).total_seconds()
            if elapsed < self.rate_limit_sec:
                import time
                time.sleep(self.rate_limit_sec - elapsed)
        self._last_request = datetime.now()

    def _parse_xml(self, xml_data: bytes) -> List[ArxivPaper]:
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as e:
            raise RuntimeError(f"Failed to parse arxiv XML: {e}")

        papers: List[ArxivPaper] = []
        for entry in root.findall("a:entry", ARXIV_NS):
            arxiv_id_raw = entry.findtext("a:id", default="", namespaces=ARXIV_NS)
            arxiv_id = arxiv_id_raw.split("/abs/")[-1] if "/abs/" in arxiv_id_raw else arxiv_id_raw
            title = " ".join(entry.findtext("a:title", default="", namespaces=ARXIV_NS).split())
            abstract = " ".join(entry.findtext("a:summary", default="", namespaces=ARXIV_NS).split())
            published = entry.findtext("a:published", default="", namespaces=ARXIV_NS)
            updated = entry.findtext("a:updated", default="", namespaces=ARXIV_NS)

            authors = []
            for author in entry.findall("a:author", ARXIV_NS):
                name = author.findtext("a:name", default="", namespaces=ARXIV_NS)
                if name:
                    authors.append(name)

            categories = []
            for cat in entry.findall("a:category", ARXIV_NS):
                term = cat.get("term")
                if term:
                    categories.append(term)

            pdf_url = ""
            for link in entry.findall("a:link", ARXIV_NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            year = 0
            try:
                year = int(published[:4])
            except (ValueError, IndexError):
                pass

            papers.append(ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published=published,
                updated=updated,
                pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
                abs_url=f"https://arxiv.org/abs/{arxiv_id}",
                year=year,
            ))
        return papers