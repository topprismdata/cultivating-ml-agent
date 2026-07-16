# 📚 表格类竞赛知识记忆库 (Knowledge Vault)

> 基于 `cultivating-ml-agent` 项目的真实竞赛经验整理  
> 最后更新：2026-07-05  
> 覆盖竞赛：Store Sales, House Prices, Spaceship Titanic, ISEC, S6E2 等

---

## 🎯 核心原则 (必须牢记)

### 1. OOF ≠ LB — 这是最重要的教训！

```
OOF (交叉验证) 提升 → LB (排行榜) 不一定提升
```

**真实案例 (S6E2)**:
| 方法 | OOF AUC | LB AUC | 结论 |
|------|---------|--------|------|
| XGBoost 基线 | 0.95551 | **0.95369** ✅ | 最优 |
| AutoGluon (100 模型) | 0.95478 | 0.95287 ❌ | 反而下降！ |
| SMOTE 30% | 0.96537 | 0.95370 | OOF 提升但 LB 持平 |
| 伪标签 | 0.96881 | 0.95368 | OOF 大幅提升但 LB 无变化 |

**教训**: 
- **永远用 LB 验证**，不要只看 OOF
- OOF 提升可能是虚假的（过拟合训练集）
- 简单模型 > 复杂模型（如果 LB 相同或更好）

---

### 2. "过犹不及" — ML Sweet Spot 原则

**更多 ≠ 更好！** 每个优化都有最优停止点：

| 优化维度 | 增加它... | 何时停止 |
|----------|-----------|----------|
| n_estimators | 改善直到平台期 | OOF 停止改善 |
| learning_rate | 越低越好（到某点） | LB 开始下降 |
| max_depth | 更深 = 更多过拟合 | 验证分下降 |
| features | 更多 = 噪声 | LB 峰值后下降 |
| seeds (集成) | 更稳定 | 边际收益递减 |

**真实案例**:
- **5-seed 集成**: OOF 0.95369 ✅
- **20-seed 集成**: OOF 0.95200 ❌ (-0.00169)
- **结论**: 3-5 个多样化模型最优，超过 10 个边际收益递减

---

### 3. CatBoost 优先策略 (2026-06-14 更新)

**当需要手动 GBDT 时，先用 CatBoost（不是 LightGBM/XGBoost）！**

**为什么 CatBoost 更好？**
1. **原生类别特征处理** — 无需手动编码（目标编码有泄漏风险）
2. **有序提升 (Ordered Boosting)** — 防止目标泄漏
3. **鲁棒的默认参数** — 大部分情况下无需调参
4. **更低的方差** — 跨种子更稳定

**验证结果 (Spaceship Titanic)**:
| 模型 | OOF 准确率 | 说明 |
|------|-----------|------|
| **CatBoost (单模型，默认)** | **0.8124** | 最佳单模型 |
| LightGBM (5 变体平均) | 0.8048 | 更差 |
| XGBoost (5 变体平均) | 0.8003 | 最差 |

**CatBoost 集成策略**:
```python
# 5 个 CatBoost 变体是甜点（不是 10+）
configs = [
    {'learning_rate': 0.02, 'depth': 6, 'l2_leaf_reg': 3.0},   # 平衡
    {'learning_rate': 0.015, 'depth': 7, 'l2_leaf_reg': 5.0},  # 更深
    {'learning_rate': 0.025, 'depth': 5, 'l2_leaf_reg': 1.0},  # 更浅
    {'learning_rate': 0.02, 'depth': 8, 'l2_leaf_reg': 4.0},   # 最深
    {'learning_rate': 0.01, 'depth': 6, 'l2_leaf_reg': 2.0},   # 最慢
]
```

**何时避免 CatBoost**:
- 超大数据集 (>1M 行) — LightGBM 更快
- 无类别特征 — CatBoost 的优势被浪费
- 需要可解释性 — CatBoost 较难解释

---

### 4. 多项式特征突破 (小数据集专用)

**适用场景**: <10K 样本，<50 特征，树模型优化遇到瓶颈

**核心思想**: 
- 树模型只捕获**轴对齐**决策边界
- 多项式特征显式捕获**交互作用** (x₁×x₂)
- 在特征空间扩展 > 模型复杂度

**验证结果 (ISEC 2026)**:
| 方法 | 特征数 | CV F1 | Public LB | Private LB | Gap |
|------|--------|-------|-----------|------------|-----|
| 基线 (16 特征) | 16 | 0.687 | 0.805 | 0.801 | 0.004 |
| Grid Search | 16 | 0.686 | 0.810 | 0.790 | 0.020 |
| **多项式 (152 特征)** | **152** | **0.710** | **0.822** | **0.812** | **0.010** ✅ |

**实现步骤**:
```python
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBClassifier

# 1. 标准化 (必须！)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 2. 生成 degree-2 多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)  # 16 → 152

# 3. 训练简单模型 (防止过拟合)
model = XGBClassifier(
    n_estimators=50,      # 减少（从 100）
    max_depth=4,          # 减少（从 6）
    learning_rate=0.1,
)
```

**何时避免**:
- 高维数据 (>50 特征) — 爆炸式增长
- 稀疏数据 — 交互放大噪声
- 可解释性关键 — 多项式项难解释

---

### 5. 最优混合策略 (80/20 规则)

**重新排序 vs 原始分数的最优混合**:

```python
blend_raw_ratio = 0.20  # 80% 重新排序 + 20% 原始

final_similarity = (
    (1 - blend_raw_ratio) * rerank_score +
    blend_raw_ratio * raw_score
)
```

**为什么 80/20?**
- 80% 重新排序：捕获邻域结构，去除假阳性
- 20% 原始：保留真实长距离相似性，维持多样性

**H-Blend 策略 (强模型主导)**:
```python
# 单个强模型应该主导
weights = {
    'model_0.944': 0.95,  # 主导！
    'model_0.938': 0.02,
    'model_0.937': 0.02,
    'model_0.930': 0.01,
}
```

---

### 6. AutoGluon 优先策略 (v0.7.0 新增)

**任何表格问题，AutoGluon 是最佳第一步（5-15 分钟）！**

**何时转手动 GBDT**:
- AutoGluon OOF 不达标
- 有 AutoGluon 抓不到的领域知识
- 想用 AutoGluon 作为 "Silver" 信号加入自定义 pipeline

**验证**: AutoGluon 在小/中等表格数据上 **2/4 战胜** 手动集成。

---

## 📋 竞赛实战 SOP (标准操作流程)

### 竞赛启动 SOP (最重要！)

```
Step 1: 数据格式验证 (kaggle-data-format-first)
   - 下载数据，检查 train/test shapes
   - 识别目标变量、评估指标
   - 检查数据类型：时间序列？图像？表格？
   - **关键**：先验证再研究，避免方向错误

Step 2: 建立知识库 (kaggle-competition-best-practices)
   - 创建 NotebookLM notebook
   - 上传 Top 方案、论坛讨论
   - 研究类似竞赛的获胜策略

Step 3: 基线模型 (AutoGluon 或 CatBoost)
   - 最简单的可行模型
   - 建立 CV 评估框架
   - **必须**：建立 CV-LB 差距的基线

Step 4: Top 方案复制 (kaggle-top-performer-replication)
   - 下载 Top notebook，逐行分析
   - 建立对比表：你的方法 vs Top 方法
   - 按预期影响力排序技术

Step 5: 系统化特征边界测试 (kaggle-feature-boundary)
   - 一次只变一个变量
   - 记录每次实验结果
   - 关注 OOF-LB 差距变化
```

---

### 模型调试 SOP (遇到问题时用)

```markdown
## 问题诊断流程

1. 检查 prediction magnitude (预测均值 vs 训练均值)
   - 如果差距 >2x → 分布不匹配问题

2. 检查 CV-LB 差距
   - CV << LB → 过拟合训练集
   - CV ≈ LB → 两者都差，模型需要改进

3. 检查特征重要性
   - Top 特征是否合理？
   - 是否有"泄露"特征？

4. 对抗验证 (adversarial-validation-implementation)
   - 检查 train/test 分布差异
   - AUC > 0.7 → 需要处理分布差异

5. 渐进式验证 (progressive-verification-debugging)
   - 从最简单的测试开始
   - 逐步增加复杂度定位问题
```

---

### 实验管理 SOP (可复现的迭代)

```markdown
## 命名规范: R{round}_{approach}

示例：R01_baseline, R02_lag7d, R05_day_specific

## 每次实验必须记录:
- CV 分数 (每个 fold 的详细分数)
- LB 分数 (提交后记录)
- 使用的特征列表
- 模型超参数
- 数据范围 (全量 or 截断)
- 特殊处理 (fillna 方式，后处理等)

## 迭代节奏:
- Day 1: 基线 + 数据验证
- Day 2: Top 方案分析 + 快速复制
- Day 3-5: 系统化改进 (特征 → 模型 → 集成)
- Day 6+: 精细调优 + 后处理
```

---

## 🚨 反模式与踩坑记录 (25+ 个)

### ML 核心反模式

| 反模式 | 症状 | 修复 |
|--------|------|------|
| **Stale Lag** | CV=0.36 LB=1.86 | Day-Specific 模型 |
| **Ensemble Correlation Trap** | 集成不如单模型 | 先检查 OOF 相关性 |
| **Per-Category Backfire** | 分类别 LB 更差 | 数据量阈值：>200K 行 |
| **Domain Constraints Trap** | 加约束后性能下降 | 检查对抗验证 AUC 变化 |
| **TE Leakage** | CV 虚高 LB 差 | Fold 内计算 TE |
| **Oil Rolling Bug** | 跨组污染 | 先按日期去重计算 |
| **fillna(0) Trap** | 模型误认为零销量 | 层级填充：sf_dow→sf→f_dow→family |
| **TTA on Pretrained** | TTA 无改善或更差 | 预训练模型不需 TTA |
| **OOF Evaluation Bug** | OOF 分数 4.0+ | 只计算非零 OOF 样本的分数 |
| **Data Truncation** | 1 年数据不如 4 年 | 除非计算限制，用全量数据 |
| **CatBoost CPU Slow** | 训练极慢 | 配置 thread_count=-1 |

### Agent 培养反模式

| 反模式 | 问题 | 修复 |
|--------|------|------|
| **过度提取** | 每个小技巧都变成技能 | 只提取非显而易见的解决方案 |
| **描述含工作流** | Agent 跳过正文走捷径 | 描述仅含触发条件 (CSO) |
| **技能重复** | 同一问题多个技能 | 5 维重叠检测，更新而非创建 |
| **叙事性写作** | "在 2026 年 2 月 3 日我们发现..." | 结构化知识，不含时间线 |
| **不验证就提取** | 理论性方案不可用 | 只提取经过验证的解决方案 |

---

## 📊 竞赛成绩追踪 (我的比赛记录)

### Store Sales Time Series Forecasting

| 版本 | CV (RMSLE) | **LB (RMSLE)** | 说明 |
|------|-----------|---------------|------|
| **R01** | 2.1045 | **1.37087** ✅ **最佳！** | 基线 LightGBM |
| R02 | 2.0503 | 1.48938 ⚠️ | Lag 7/14/30d (错误实现) |
| R05 (v1) | 3.3962 | 2.13508 ⚠️ | Day-Specific (简化) |
| R05 (v2) | 3.3962 | 2.09960 ✅ 改进 | Day-Specific (16 模型) |
| R05 (v3) | 2.3336 | 2.65512 ❌ 更差 | Day-Specific + 增强特征 |
| **Top 10%** | ~0.39-0.42 | - | 竞赛目标 |

**关键发现**:
1. **R01 仍然是最好的** — 简单基线模型在测试集上最鲁棒
2. **Day-Specific 方法本身有效**（R05 v2 = LB 2.10）
3. **复杂特征工程在测试集上容易引入噪声**（R05 Enhanced = LB 2.66）
4. **OOF ≠ LB** — 这是贯穿始终的教训

---

## 🎓 学术对齐 (研究基础)

| 学术概念 | 我们的实践 |
|---------|----------|
| AIDE (Huang 2024) trial-and-error | 详细失败记录 |
| AutoMind (Zhang 2025) 知识库 | 120+ SKILL.md 三层架构 |
| Voyager (Wang 2023) skill library | Claudeception 自动提取 |
| CoMind (2025) 记忆架构 | 全局/项目/skill 三层记忆 |
| Reflexion (Shinn 2023) 经验反思 | 三层智慧提取 |
| NFD (Zhang 2026) 培养优先 | 核心哲学 |
| **AutoGluon (Fakoor 2020)** | **表格比赛第一步** |
| **TabPFN (Hollmann 2023)** | **小表格 Transformer（未来方向）** |

---

## 📝 使用指南

### 何时查阅此文档

1. **开始新竞赛时** — 阅读"竞赛启动 SOP"
2. **遇到 CV-LB 差距时** — 检查"模型调试 SOP"和"反模式"
3. **选择模型时** — 参考"CatBoost 优先策略"和"多项式特征"
4. **集成模型时** — 参考"最优混合策略"和"Sweet Spot 原则"

### 如何更新此文档

每次竞赛结束后：
1. 将新发现添加到"竞赛成绩追踪"
2. 将新的反模式添加到"反模式与踩坑记录"
3. 更新相关 SOP（如果有新发现）

---

## 🔗 相关文件

- `skills/examples/` — 28+ 个真实 skills
- `docs/cultivating-ml-agent-expert.md` — 主指南 (1291 行)
- `AGENTS.md` — Agent 自主工作流指令

---

**最后更新**: 2026-07-05  
**版本**: 1.0  
**状态**: 持续更新中

---

*基于 [cultivating-ml-agent](https://github.com/topprismdata/cultivating-ml-agent) 项目整理*
