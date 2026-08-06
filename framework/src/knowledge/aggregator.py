"""
knowledge aggregator — 统一搜索接口

把 arxiv / Kaggle / Semantic Scholar 三个数据源整合,
自动去重、按相关性排序、可选写入 OKF 知识图谱。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterable
from pathlib import Path

from .arxiv_search import ArxivSearch, ArxivPaper
from .kaggle_forum import KaggleForum, KaggleDiscussion
from .semantic_scholar import SemanticScholar, SSPaper


@dataclass
class KnowledgeReport:
    """一次搜索的聚合报告"""
    query: str
    arxiv_papers: List[ArxivPaper] = field(default_factory=list)
    kaggle_discs: List[KaggleDiscussion] = field(default_factory=list)
    scholar_papers: List[SSPaper] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return len(self.arxiv_papers) + len(self.kaggle_discs) + len(self.scholar_papers)

    def to_markdown(self) -> str:
        lines = [f"# Knowledge Report: \"{self.query}\"\n"]
        lines.append(f"**Total results**: {self.total_count} "
                     f"(arxiv: {len(self.arxiv_papers)}, "
                     f"kaggle: {len(self.kaggle_discs)}, "
                     f"scholar: {len(self.scholar_papers)})\n\n")

        if self.scholar_papers:
            lines.append("## 📚 Semantic Scholar (高引用优先)\n\n")
            for p in self.scholar_papers[:10]:
                lines.append(p.to_markdown())

        if self.arxiv_papers:
            lines.append("## 🔬 arxiv (最新)\n\n")
            for p in self.arxiv_papers[:10]:
                lines.append(p.to_markdown())

        if self.kaggle_discs:
            lines.append("## 💬 Kaggle Discussions (实战经验)\n\n")
            for d in self.kaggle_discs[:10]:
                lines.append(d.to_markdown())

        return "".join(lines)


class KnowledgeAggregator:
    """聚合三个数据源,提供统一搜索"""

    def __init__(self,
                 arxiv: Optional[ArxivSearch] = None,
                 kaggle: Optional[KaggleForum] = None,
                 scholar: Optional[SemanticScholar] = None,
                 okf_dir: Optional = None):
        self.arxiv = arxiv or ArxivSearch()
        self.kaggle = kaggle or KaggleForum()
        self.scholar = scholar or SemanticScholar()
        self.okf_dir = okf_dir  # 可选,查询结果可写入

    def search_all(self, query: str,
                   max_per_source: int = 10,
                   years_back: int = 5) -> KnowledgeReport:
        """同时在三个数据源搜索"""
        report = KnowledgeReport(query=query)
        errors = []

        # 1. Semantic Scholar(优先,带 TLDR 和引用)
        try:
            report.scholar_papers = self.scholar.search(
                query,
                max_results=max_per_source,
                year_range=(2026 - years_back, 2026),
            )
        except Exception as e:
            errors.append(f"Semantic Scholar: {e}")

        # 2. arxiv(最新论文)
        try:
            report.arxiv_papers = self.arxiv.search(
                query,
                max_results=max_per_source,
                category="cs.LG",
                years_back=min(years_back, 5),
            )
        except Exception as e:
            errors.append(f"arxiv: {e}")

        # 3. Kaggle(实战讨论)
        try:
            report.kaggle_discs = self.kaggle.search_global(query, max_results=max_per_source)
        except Exception as e:
            errors.append(f"Kaggle: {e}")

        if errors:
            report.metadata_errors = errors

        return report

    def search_papers_only(self, query: str,
                           max_results: int = 20,
                           years_back: int = 5) -> List[SSPaper]:
        """只搜论文(arxiv + Semantic Scholar,合并去重)"""
        papers = {}

        # Semantic Scholar 优先
        try:
            for p in self.scholar.search(query, max_results=max_results,
                                          year_range=(2026 - years_back, 2026)):
                papers[p.paper_id or p.title] = p
        except Exception:
            pass

        # arxiv 补全
        try:
            for p in self.arxiv.search(query, max_results=max_results,
                                        category="cs.LG",
                                        years_back=years_back):
                key = p.arxiv_id or p.title
                if key not in papers:
                    papers[key] = p
        except Exception:
            pass

        return list(papers.values())[:max_results]

    def write_report_to_vault(self, report: KnowledgeReport,
                              target_dir: Optional = None) -> Path:
        """把报告写入 OKF vault(便于以后检索)"""
        target_dir = Path(target_dir or self.okf_dir or "docs/ml-agent-memory")
        target_dir = target_dir / "auto-search"
        target_dir.mkdir(parents=True, exist_ok=True)

        # 文件名基于 query + timestamp
        from datetime import datetime
        import re
        safe_query = re.sub(r"[^a-z0-9]+", "-", report.query.lower())[:50]
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = target_dir / f"{safe_query}-{ts}.md"

        # 写 frontmatter
        frontmatter = (
            "---\n"
            f"query: \"{report.query}\"\n"
            f"total: {report.total_count}\n"
            f"arxiv: {len(report.arxiv_papers)}\n"
            f"kaggle: {len(report.kaggle_discs)}\n"
            f"scholar: {len(report.scholar_papers)}\n"
            f"timestamp: {ts}\n"
            "type: auto-search\n"
            "importance: 0.5\n"
            "tags: auto,arxiv,kaggle,scholar\n"
            "---\n\n"
        )
        path.write_text(frontmatter + report.to_markdown(), encoding="utf-8")
        return path