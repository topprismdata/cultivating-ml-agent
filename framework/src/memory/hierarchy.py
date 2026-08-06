"""
3 层记忆管理 — 主实现

架构(MemGPT 风格):
    Working Memory     主上下文,容量有限,LRU 淘汰
    Archival Store     OKF 图谱索引,无限容量,按需检索
    Long-term Storage  filesystem 沉淀层

为什么:
    Skills(43+)塞进主上下文会爆;但每次召回又太慢。
    解决方案:分层 + 相关性召回,只把"当下相关"的 items 拉到 working。
"""
from __future__ import annotations

import re
import os
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Iterable, Set


# -----------------------------
# MemoryItem: 记忆的原子单位
# -----------------------------

@dataclass
class MemoryItem:
    """统一表示 skill / principle / anti-pattern / experiment / session 笔记"""
    id: str
    content: str
    type: str = "skill"   # "skill" | "principle" | "anti-pattern" | "experiment" | "session"
    importance: float = 0.5  # 0-1, ≥0.7 自动晋升到 archival
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    last_accessed: str = ""
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def touch(self):
        self.last_accessed = datetime.now().isoformat(timespec="seconds")
        self.access_count += 1

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "MemoryItem":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# -----------------------------
# Working Memory — 主上下文层
# -----------------------------

class WorkingMemory:
    """LRU 缓存式主上下文。容量满了自动淘汰最久未用。"""

    DEFAULT_MAX = 20  # context window 友好值

    def __init__(self, max_items: int = DEFAULT_MAX):
        self.max_items = max_items
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem):
        # 去重
        self.items = [i for i in self.items if i.id != item.id]
        item.touch()
        self.items.append(item)
        if len(self.items) > self.max_items:
            self._evict_lru()

    def get(self, item_id: str) -> Optional[MemoryItem]:
        for item in self.items:
            if item.id == item_id:
                item.touch()
                return item
        return None

    def remove(self, item_id: str) -> bool:
        before = len(self.items)
        self.items = [i for i in self.items if i.id != item_id]
        return len(self.items) < before

    def _evict_lru(self):
        if not self.items:
            return
        lru = min(self.items, key=lambda x: x.last_accessed or x.created_at)
        self.items.remove(lru)

    def context_snapshot(self, max_chars: int = 8000) -> str:
        """渲染成可塞进 prompt 的纯文本"""
        lines = [f"# Working Memory ({len(self.items)}/{self.max_items} items)\n"]
        chars = len(lines[0])
        for item in self.items:
            block = f"\n## [{item.type}] {item.id} (imp={item.importance:.2f}, accessed={item.access_count}x)\n{item.content}\n"
            if chars + len(block) > max_chars:
                lines.append(f"\n... ({len(self.items) - len(lines) + 1} more items truncated)\n")
                break
            lines.append(block)
            chars += len(block)
        return "".join(lines)


# -----------------------------
# Archival Store — OKF 索引层
# -----------------------------

class ArchivalStore:
    """OKF markdown + filesystem 索引。简单 TF 评分,够用,可升级为 embedding。"""

    def __init__(self, okf_dir: str = "docs/ml-agent-memory"):
        self.okf_dir = Path(okf_dir)
        self.items: Dict[str, MemoryItem] = {}

    def index_directory(self, extra_dirs: Optional[Iterable[str]] = None) -> int:
        """扫描 OKF 目录,提取所有 .md 文件作为 memory item。返回索引条数。"""
        paths = []
        if self.okf_dir.exists():
            paths.extend(self.okf_dir.rglob("*.md"))
        for d in extra_dirs or []:
            p = Path(d)
            if p.exists():
                paths.extend(p.rglob("*.md"))
        for path in paths:
            item = self._md_to_item(path)
            if item:
                self.items[item.id] = item
        return len(self.items)

    def add(self, item: MemoryItem):
        self.items[item.id] = item

    def search(self, query: str, k: int = 5,
               type_filter: Optional[str] = None,
               min_importance: float = 0.0) -> List[MemoryItem]:
        """TF + importance 评分"""
        if not query.strip():
            return []
        q_terms = set(self._tokenize(query))
        if not q_terms:
            return []

        scored: List[tuple[float, MemoryItem]] = []
        for item in self.items.values():
            if type_filter and item.type != type_filter:
                continue
            if item.importance < min_importance:
                continue
            c_terms = set(self._tokenize(item.content))
            if not c_terms:
                continue
            overlap = len(q_terms & c_terms)
            if overlap == 0:
                continue
            # TF score * importance * (access_count 平滑)
            score = (overlap / math.sqrt(len(q_terms) * len(c_terms))) \
                    * (0.5 + item.importance) \
                    * (1 + math.log1p(item.access_count))
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    # ---- helpers ----

    def _md_to_item(self, path: Path) -> Optional[MemoryItem]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        # 解析 YAML frontmatter(如果有)
        meta: Dict = {}
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                fm_block = parts[1]
                body = parts[2]
                for line in fm_block.splitlines():
                    if ":" in line:
                        k, _, v = line.partition(":")
                        meta[k.strip()] = v.strip()
                content = body

        rel = path.relative_to(self.okf_dir) if self.okf_dir in path.parents else path.name
        item_id = str(rel).replace("\\", "/").replace(".md", "")
        item_type = self._infer_type(str(path), meta)
        importance = float(meta.get("importance", 0.6))
        tags = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
        return MemoryItem(
            id=item_id,
            content=content.strip(),
            type=item_type,
            importance=importance,
            tags=tags,
            metadata={"path": str(path), "title": meta.get("title", path.stem)},
        )

    def _infer_type(self, path_str: str, meta: Dict) -> str:
        if "type" in meta:
            return meta["type"]
        p = path_str.lower()
        if "skill" in p or "/skills/" in p:
            return "skill"
        if "principle" in p or "/principles/" in p:
            return "principle"
        if "anti-pattern" in p:
            return "anti-pattern"
        if "experiment" in p:
            return "experiment"
        return "session"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # 简单英文/中文混合分词(对中文按字)
        text = text.lower()
        # 抽取英文单词 + 单个中文字
        tokens = re.findall(r"[a-z][a-z0-9_]+|[\u4e00-\u9fff]", text)
        return [t for t in tokens if len(t) > 1 or t in {"i", "a"}]


# -----------------------------
# MemoryHierarchy — 3 层协调器
# -----------------------------

class MemoryHierarchy:
    """3 层记忆管理器:working ↔ archival ↔ filesystem"""

    def __init__(self, okf_dir: str = "docs/ml-agent-memory",
                 working_capacity: int = 20):
        self.working = WorkingMemory(max_items=working_capacity)
        self.archival = ArchivalStore(okf_dir=okf_dir)
        self.indexed_count = 0

    def bootstrap(self, extra_dirs: Optional[Iterable[str]] = None) -> int:
        """启动时索引 OKF 目录"""
        self.indexed_count = self.archival.index_directory(extra_dirs)
        return self.indexed_count

    def recall(self, query: str, k: int = 5,
               auto_promote: bool = True) -> List[MemoryItem]:
        """检索 + 自动晋升到 working"""
        hits = self.archival.search(query, k=k)
        if auto_promote:
            for item in hits:
                if not self.working.get(item.id):
                    self.working.add(item)
        return hits

    def remember(self, item: MemoryItem, persist: bool = False):
        """新增一条记忆。importance ≥0.7 自动晋升到 archival"""
        self.working.add(item)
        if item.importance >= 0.7 or persist:
            self.archival.add(item)
            if persist:
                self._persist_item(item)

    def forget(self, item_id: str) -> bool:
        """从 working 移除(不影响 archival)"""
        return self.working.remove(item_id)

    def snapshot(self, max_chars: int = 8000) -> str:
        """主上下文快照"""
        return self.working.context_snapshot(max_chars=max_chars)

    def stats(self) -> Dict:
        return {
            "working_items": len(self.working.items),
            "working_capacity": self.working.max_items,
            "archival_items": len(self.archival.items),
            "indexed_count": self.indexed_count,
        }

    def _persist_item(self, item: MemoryItem):
        """写入 filesystem(简单实现,生产可换 OKF 规范)"""
        target = self.archival.okf_dir / "auto" / f"{item.id.replace('/', '_')}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        front = "---\n" + json.dumps({
            "type": item.type,
            "importance": item.importance,
            "tags": ",".join(item.tags),
            "created_at": item.created_at,
        }, indent=2, ensure_ascii=False) + "\n---\n\n"
        target.write_text(front + item.content, encoding="utf-8")


# -----------------------------
# 便捷函数:与现有 framework 集成
# -----------------------------

def recall_skills_for_query(query: str, k: int = 5,
                            okf_dir: str = "docs/ml-agent-memory",
                            skills_dir: str = "skills/examples") -> List[MemoryItem]:
    """召回与 query 最相关的 skills。在 agent 决策点调用。"""
    arch = ArchivalStore(okf_dir=okf_dir)
    arch.index_directory(extra_dirs=[skills_dir])
    return arch.search(query, k=k, type_filter="skill")