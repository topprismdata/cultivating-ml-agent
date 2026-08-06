"""
OKF 索引 — YAML frontmatter 解析 + 知识图谱边遍历

依赖项目已有的 OKF 格式(见 docs/okf-migration-report.md)。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Set

from .hierarchy import MemoryItem, ArchivalStore


class OKFEdge:
    """知识图谱边:从一个概念节点指向另一个"""
    def __init__(self, source: str, target: str, relation: str, weight: float = 1.0):
        self.source = source
        self.target = target
        self.relation = relation   # "exemplifies" | "contradicts" | "uses" | "extends" | "related"
        self.weight = weight

    def __repr__(self):
        return f"{self.source} --{self.relation}--> {self.target}"


class OKFIndex:
    """OKF 知识图谱索引 — 提取 frontmatter + markdown body 中的引用边"""

    # 常见引用模式
    EDGE_PATTERNS = [
        # markdown links: [text](path.md) 或 [text](path.md#section)
        (re.compile(r"\[([^\]]+)\]\(([^\)]+\.md)(?:#[^\)]+)?\)"), "related"),
        # 显式反引号引用: `path/to/file.md`
        (re.compile(r"`([a-z0-9_\-/]+\.md)`", re.IGNORECASE), "uses"),
        # See also 模式
        (re.compile(r"[Ss]ee also:?\s*`?([a-z0-9_\-/]+\.md)`?", re.IGNORECASE), "related"),
        # "extends" 模式
        (re.compile(r"[Ee]xtends:?\s*`?([a-z0-9_\-/]+\.md)`?", re.IGNORECASE), "extends"),
        # "contradicts" 模式
        (re.compile(r"[Cc]ontradicts:?\s*`?([a-z0-9_\-/]+\.md)`?", re.IGNORECASE), "contradicts"),
    ]

    def __init__(self, okf_dir: str = "docs/ml-agent-memory"):
        self.okf_dir = Path(okf_dir)
        self.items: Dict[str, MemoryItem] = {}
        self.edges: List[OKFEdge] = []
        self._edges_by_source: Dict[str, List[OKFEdge]] = {}

    def build(self, extra_dirs: Optional[Iterable[str]] = None) -> int:
        """构建索引:扫描所有 .md,提取 frontmatter + 边"""
        paths = list(self.okf_dir.rglob("*.md")) if self.okf_dir.exists() else []
        for d in extra_dirs or []:
            p = Path(d)
            if p.exists():
                paths.extend(p.rglob("*.md"))

        for path in paths:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            meta, body = self._parse_frontmatter(content)
            item_id = self._make_id(path)
            item = MemoryItem(
                id=item_id,
                content=body.strip(),
                type=meta.get("type", self._infer_type(str(path))),
                importance=float(meta.get("importance", 0.6)),
                tags=[t.strip() for t in meta.get("tags", "").split(",") if t.strip()],
                metadata={"path": str(path), "title": meta.get("title", path.stem)},
            )
            self.items[item_id] = item

            # 提取边
            for edge in self._extract_edges(item_id, content):
                self.edges.append(edge)
                self._edges_by_source.setdefault(edge.source, []).append(edge)

        return len(self.items)

    def neighbors(self, item_id: str, relation: Optional[str] = None) -> List[str]:
        """获取相邻节点"""
        edges = self._edges_by_source.get(item_id, [])
        if relation:
            edges = [e for e in edges if e.relation == relation]
        return list({e.target for e in edges})

    def reverse_neighbors(self, item_id: str) -> List[str]:
        """谁指向我?"""
        return list({e.source for e in self.edges if e.target == item_id})

    def expand_query(self, item_ids: List[str], depth: int = 1) -> List[str]:
        """从起始 nodes 出发,广度优先扩展 N 层"""
        visited: Set[str] = set(item_ids)
        frontier = list(item_ids)
        for _ in range(depth):
            next_frontier = []
            for nid in frontier:
                for n in self.neighbors(nid):
                    if n not in visited:
                        visited.add(n)
                        next_frontier.append(n)
            frontier = next_frontier
        return list(visited)

    # ---- helpers ----

    def _parse_frontmatter(self, content: str) -> tuple[Dict, str]:
        meta: Dict = {}
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                for line in parts[1].splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        meta[k.strip()] = v.strip()
                return meta, parts[2]
        return meta, content

    def _make_id(self, path: Path) -> str:
        try:
            rel = path.relative_to(self.okf_dir)
            return str(rel).replace("\\", "/").replace(".md", "")
        except ValueError:
            return path.stem

    def _infer_type(self, path_str: str) -> str:
        p = path_str.lower()
        if "/skills/" in p:
            return "skill"
        if "/principles/" in p:
            return "principle"
        if "/anti-patterns/" in p:
            return "anti-pattern"
        if "/experiments/" in p:
            return "experiment"
        return "session"

    def _extract_edges(self, source_id: str, content: str) -> Iterable[OKFEdge]:
        edges: List[OKFEdge] = []
        for pattern, relation in self.EDGE_PATTERNS:
            for m in pattern.finditer(content):
                target = m.group(2) if pattern.pattern.startswith(r"\[") else m.group(1)
                # 去掉 .md 后缀归一化
                target = target.replace(".md", "").strip("/")
                if target and target != source_id:
                    edges.append(OKFEdge(source_id, target, relation))
        return edges