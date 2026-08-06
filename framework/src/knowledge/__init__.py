"""
knowledge package — 实时知识获取(论文 + Kaggle 论坛 + Semantic Scholar)
"""
from .arxiv_search import ArxivSearch, ArxivPaper
from .kaggle_forum import KaggleForum, KaggleDiscussion
from .semantic_scholar import SemanticScholar, SSPaper
from .aggregator import KnowledgeAggregator, KnowledgeReport

__all__ = [
    "ArxivSearch", "ArxivPaper",
    "KaggleForum", "KaggleDiscussion",
    "SemanticScholar", "SSPaper",
    "KnowledgeAggregator", "KnowledgeReport",
]