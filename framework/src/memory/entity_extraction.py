"""
Entity-Relation Extraction — Mem0/Letta 风格的动态图记忆

agy 建议:升级静态 OKF 到 Entity-Relation Extraction(自动从文本提取概念和关系)

设计:
    - 启发式:基于模式匹配的实体抽取(无需 LLM 即可工作)
    - 可选:LLM 增强(用 call_llm 注入)

升级点:未来可接 [GLiNER](https://github.com/urchade/GLiNER) 或 LLM-based NER
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable


@dataclass
class Entity:
    """抽取出的实体"""
    name: str
    type: str            # "concept" | "technique" | "tool" | "metric" | "competition" | "person"
    aliases: Set[str] = field(default_factory=set)
    mentions: int = 0

    def __repr__(self):
        return f"Entity({self.name}, {self.type})"


@dataclass
class Relation:
    """实体间的关系"""
    source: str
    target: str
    relation: str        # "uses" | "extends" | "contradicts" | "improves" | "enables" | "related"
    confidence: float = 1.0
    evidence: str = ""   # 原文片段

    def __repr__(self):
        return f"Relation({self.source} --{self.relation}--> {self.target})"


# 实体类型关键词(基于 ML 领域)
ENTITY_TYPE_PATTERNS = {
    "technique": re.compile(
        r"\b(LightGBM|XGBoost|CatBoost|AutoGluon|Random Forest|Neural Network|"
        r"Transformer|LSTM|GAN|CNN|RNN|ResNet|BERT|GPT|TabPFN|"
        r"walk[- ]forward|time[- ]series|target encoding|feature engineering|"
        r"ensemble|knowledge distillation|fine[- ]tuning|"
        r"backpropagation|gradient descent|优化|特征工程|集成学习|交叉验证)\b",
        re.IGNORECASE
    ),
    "tool": re.compile(
        r"\b(pandas|numpy|scikit-learn|sklearn|pytorch|tensorflow|jupyter|"
        r"mlflow|optuna|wandb|kaggle|github|arxiv|scipy|matplotlib|"
        r"seaborn|plotly|streamlit|django|fastapi|docker|kubernetes)\b",
        re.IGNORECASE
    ),
    "metric": re.compile(
        r"\b(F1|AUC|RMSE|MAE|RMSLE|accuracy|precision|recall|"
        r"log[- ]loss|NDCG|MAPE|MSE|R²|BLEU|perplexity)\b",
        re.IGNORECASE
    ),
    "competition": re.compile(
        r"\b(Kaggle|Titanic|House Prices|Spaceship Titanic|Store Sales|"
        r"M5|M4|Favorita|Instacart|IEEE[- ]CIS|Jane Street|"
        r"S6E\d+|S5E\d+)\b",
        re.IGNORECASE
    ),
    "concept": re.compile(
        r"\b(overfitting|underfitting|data leakage|distribution shift|"
        r"concept drift|cold[- ]start|warm[- ]start|baseline|"
        r"hyperparameter|epoch|batch[- ]size|learning[- ]rate|"
        r"regularization|dropout|batch[- ]norm|"
        r"过拟合|欠拟合|数据泄露|分布偏移)\b",
        re.IGNORECASE
    ),
}

# 关系模式
RELATION_PATTERNS = [
    # (regex, relation_type)
    (re.compile(r"(\w+)\s+(?:uses|uses? to)\s+(\w+)", re.IGNORECASE), "uses"),
    (re.compile(r"(\w+)\s+(?:improves|better than|beats)\s+(\w+)", re.IGNORECASE), "improves"),
    (re.compile(r"(\w+)\s+(?:extends?|builds? on)\s+(\w+)", re.IGNORECASE), "extends"),
    (re.compile(r"(\w+)\s+(?:enables?|allows?)\s+(\w+)", re.IGNORECASE), "enables"),
    (re.compile(r"(\w+)\s+(?:vs\.?|versus|compared to)\s+(\w+)", re.IGNORECASE), "related"),
    (re.compile(r"(\w+)\s+(?:contradicts?|conflicts? with)\s+(\w+)", re.IGNORECASE), "contradicts"),
    (re.compile(r"(\w+)\s*->\s*(\w+)", re.IGNORECASE), "related"),
]


class EntityRelationExtractor:
    """实体-关系抽取器"""

    def __init__(self, llm_call: Optional[Callable] = None):
        """
        Args:
            llm_call: 可选 LLM 调用函数,用于增强抽取
        """
        self.llm_call = llm_call

    def extract(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """从文本抽取实体和关系

        Returns:
            (entities, relations)
        """
        # 1. 实体抽取
        entities = self._extract_entities(text)

        # 2. 关系抽取
        relations = self._extract_relations(text, entities)

        # 3. 可选 LLM 增强
        if self.llm_call:
            extra = self._llm_enhance(text, entities, relations)
            entities.extend(extra.get("entities", []))
            relations.extend(extra.get("relations", []))

        # 去重
        entities = self._dedup_entities(entities)
        relations = self._dedup_relations(relations)
        return entities, relations

    def _extract_entities(self, text: str) -> List[Entity]:
        """基于模式匹配的实体抽取"""
        entities: Dict[str, Entity] = {}
        for entity_type, pattern in ENTITY_TYPE_PATTERNS.items():
            for m in pattern.finditer(text):
                name = m.group(0).strip()
                key = name.lower()
                if key not in entities:
                    entities[key] = Entity(name=name, type=entity_type)
                entities[key].mentions += 1
        return list(entities.values())

    def _extract_relations(self, text: str, entities: List[Entity]
                           ) -> List[Relation]:
        """基于模式匹配的关系抽取"""
        entity_names = {e.name.lower() for e in entities}
        relations: List[Relation] = []

        for pattern, rel_type in RELATION_PATTERNS:
            for m in pattern.finditer(text):
                source = m.group(1).strip()
                target = m.group(2).strip()
                # 只保留出现在 entities 中的关系
                if source.lower() in entity_names and target.lower() in entity_names:
                    relations.append(Relation(
                        source=source,
                        target=target,
                        relation=rel_type,
                        evidence=text[max(0, m.start() - 30):min(len(text), m.end() + 30)],
                    ))
        return relations

    def _llm_enhance(self, text: str, entities: List[Entity],
                     relations: List[Relation]) -> Dict:
        """LLM 增强(可选)"""
        if not self.llm_call:
            return {"entities": [], "relations": []}
        prompt = (
            "Extract ML entities and relations from the text below.\n"
            "Return as JSON: {\"entities\": [{\"name\":, \"type\":}], "
            "\"relations\": [{\"source\":, \"target\":, \"relation\":}]}\n\n"
            f"Text: {text[:2000]}"
        )
        try:
            import json
            response = self.llm_call(prompt)
            data = json.loads(response)
            return {
                "entities": [Entity(**e) for e in data.get("entities", [])],
                "relations": [Relation(**r) for r in data.get("relations", [])],
            }
        except Exception:
            return {"entities": [], "relations": []}

    def _dedup_entities(self, entities: List[Entity]) -> List[Entity]:
        """去重实体(按 name 小写)"""
        seen: Dict[str, Entity] = {}
        for e in entities:
            key = e.name.lower()
            if key in seen:
                seen[key].mentions += e.mentions
                seen[key].aliases.update(e.aliases)
            else:
                seen[key] = e
        return list(seen.values())

    def _dedup_relations(self, relations: List[Relation]) -> List[Relation]:
        """去重关系"""
        seen = set()
        unique = []
        for r in relations:
            key = (r.source.lower(), r.target.lower(), r.relation)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique