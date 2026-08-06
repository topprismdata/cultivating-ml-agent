"""
Semantic Scholar 论文搜索 — arxiv 的补充

优势:
    - 引用图谱(谁引用了谁)
    - TLDR 自动摘要
    - 影响力评分

API: https://api.semanticscholar.org/graph/v1/paper/search
免费、无需 key(有速率限制,生产环境建议申请 key)
"""
from __future__ import annotations

import urllib.request
import urllib.parse
import json
from dataclasses import dataclass, field
from typing import List, Optional


SS_API = "https://api.semanticscholar.org/graph/v1"


@dataclass
class SSPaper:
    """一篇 Semantic Scholar 论文"""
    paper_id: str
    title: str
    abstract: Optional[str]
    year: Optional[int]
    venue: str
    citation_count: int
    reference_count: int
    influential_citation_count: int
    authors: List[str]
    url: str
    tldr: Optional[str] = None
    fields_of_study: List[str] = field(default_factory=list)

    def __repr__(self):
        first_author = self.authors[0] if self.authors else "?"
        return f"[{self.year or '?'}] {self.title} — {first_author} ({self.citation_count} cites)"

    def to_markdown(self) -> str:
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += f" ... +{len(self.authors) - 3}"
        tldr_block = f"\n- **TLDR**: {self.tldr}" if self.tldr else ""
        return (
            f"## {self.title}\n\n"
            f"- **Authors**: {authors_str}\n"
            f"- **Year**: {self.year}\n"
            f"- **Venue**: {self.venue}\n"
            f"- **Citations**: {self.citation_count} (influential: {self.influential_citation_count})\n"
            f"- **Fields**: {', '.join(self.fields_of_study[:5])}\n"
            f"- **URL**: {self.url}\n"
            f"{tldr_block}\n"
            f"- **Abstract**: {(self.abstract or '')[:500]}...\n\n"
        )


class SemanticScholar:
    """Semantic Scholar 论文搜索客户端"""

    def __init__(self, api_key: Optional = None, rate_limit_sec: float = 1.0):
        """
        Args:
            api_key: 可选 API key(申请:https://www.semanticscholar.org/product/api)
                   无 key 时 100 req/5min,有 key 时提升到 5000 req/5min
            rate_limit_sec: 请求间隔
        """
        self.api_key = api_key
        self.rate_limit_sec = rate_limit_sec
        self._last_request = None

    def search(self, query: str, max_results: int = 20,
               year_range: Optional = None,
               fields_of_study: Optional = None) -> List[SSPaper]:
        """搜索论文

        Args:
            query: 搜索关键词
            max_results: 最大返回数(API 上限 100)
            year_range: 年份范围 tuple (start, end)
            fields_of_study: 限定领域,["Computer Science", "Machine Learning"]
        """
        self._respect_rate_limit()

        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": ",".join([
                "title", "abstract", "year", "venue", "citationCount",
                "referenceCount", "influentialCitationCount", "authors",
                "url", "tldr", "fieldsOfStudy",
            ]),
        }
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        url = f"{SS_API}/paper/search?" + urllib.parse.urlencode(params)
        headers = {"User-Agent": "cultivating-ml-agent/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception as e:
            raise RuntimeError(f"Semantic Scholar request failed: {e}")

        return [self._parse_paper(p) for p in data.get("data", [])]

    def get_citations(self, paper_id: str, max_results: int = 50) -> List[SSPaper]:
        """获取某篇论文的引用列表(谁引用了它)"""
        self._respect_rate_limit()
        url = (
            f"{SS_API}/paper/{urllib.parse.quote(paper_id)}/citations"
            f"?fields=title,authors,year,venue,citationCount,url&limit={max_results}"
        )
        headers = {"User-Agent": "cultivating-ml-agent/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8", errors="replace"))
            return [self._parse_paper(c.get("citingPaper", {})) for c in data.get("data", [])]
        except Exception:
            return []

    def get_references(self, paper_id: str, max_results: int = 50) -> List[SSPaper]:
        """获取某篇论文的参考文献列表"""
        self._respect_rate_limit()
        url = (
            f"{SS_API}/paper/{urllib.parse.quote(paper_id)}/references"
            f"?fields=title,authors,year,venue,citationCount,url&limit={max_results}"
        )
        headers = {"User-Agent": "cultivating-ml-agent/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8", errors="replace"))
            return [self._parse_paper(r.get("citedPaper", {})) for r in data.get("data", [])]
        except Exception:
            return []

    # ---- helpers ----

    def _parse_paper(self, p: dict) -> SSPaper:
        authors = [a.get("name", "") for a in p.get("authors", []) if a.get("name")]
        tldr_obj = p.get("tldr") or {}
        return SSPaper(
            paper_id=p.get("paperId", ""),
            title=p.get("title", "(no title)"),
            abstract=p.get("abstract"),
            year=p.get("year"),
            venue=p.get("venue", ""),
            citation_count=p.get("citationCount", 0) or 0,
            reference_count=p.get("referenceCount", 0) or 0,
            influential_citation_count=p.get("influentialCitationCount", 0) or 0,
            authors=authors,
            url=p.get("url", ""),
            tldr=tldr_obj.get("text") if isinstance(tldr_obj, dict) else None,
            fields_of_study=p.get("fieldsOfStudy", []) or [],
        )

    def _respect_rate_limit(self):
        if self._last_request:
            import time
            elapsed = time.time() - self._last_request
            if elapsed < self.rate_limit_sec:
                time.sleep(self.rate_limit_sec - elapsed)
        import time as t
        self._last_request = t.time()