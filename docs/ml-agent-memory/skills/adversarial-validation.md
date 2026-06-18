---
type: Skill
title: Adversarial Validation 决策框架
description: 'Adversarial Validation 决策框架：识别与测试集最相似的训练样本。'
tags: [skill, adversarial-validation, diagnostic]
timestamp: 2026-06-15T10:05:55Z
---

# Adversarial Validation 决策框架

> 来源: `adversarial-validation-implementation` skill + v16 实验教训

---

## 核心原则

**目标**: 识别与测试集最相似的训练样本，用于模型训练

**常见误解**:
- ❌ "对抗验证区分真实 vs 合成数据"
- ✅ "对抗验证区分训练 vs 测试分布"

---

## 决策树

```
开始
  │
  ├─ 任务类型是 Kaggle Playground Series?
  │     └─ YES → 使用对抗验证
  │           │
  │           └─ NO → 可能不需要（数据已对齐）
  │
  ├─ CV-LB gap > 0.01?
  │     └─ YES → 使用对抗验证
  │           │
  │           └─ NO → 可能不需要
  │
  └─ 数据集 > 100K 且怀疑 GAN artifacts?
        └─ YES → 使用对抗验证
```

---

## 实现步骤

### Step 1: 构建对抗数据集
```python
# Combine train and test sets
adv_train = train[features].copy()
adv_train['is_test'] = 0  # 训练集标签

adv_test = test[features].copy()
adv_test['is_test'] = 1  # 测试集标签

adv_combined = pd.concat([adv_train, adv_test], axis=0, ignore_index=True)
```

### Step 2: 训练对抗分类器
```python
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'verbosity': -1,
    'n_jobs': 10
}
```

### Step 3: AUC 解读
| AUC | 含义 | 行动 |
|-----|------|------|
| ≈ 0.50 | train/test 已对齐 | **停止净化**，探索其他方向 |
| 0.50-0.60 | 轻微漂移 | 保守净化（80% 数据）|
| 0.60-0.70 | 中等漂移 | 适度净化（40-60% 数据）|
| > 0.70 | 显著漂移 | 积极净化（<30% 数据）|

---

## S6E5 v16 实验教训

**发现**: Adversarial AUC = 0.50309 ≈ 0.50

**结论**:
1. train/test 分布已完全对齐
2. 净化过滤不改善（60K 净化 ≈ 439K 完整，CV 相同）
3. CV-LB gap (~0.01) **不是由分布不匹配造成**

**新原则**:
- Adversarial AUC ≈ 0.50 → 立即转向其他方向
- 不要再浪费时间在净化策略上

---

## 反面模式

| 错误方法 | 正确方法 |
|---------|---------|
| 区分 train vs UCI (真实数据) | 区分 train vs test |
| 百分位过滤 | 排序后取前 N 个 |
| 先净化再做 TE | 先加外部数据再做 TE |

---

## 特征重要性 → 诊断信号

**S6E5 Top 区分特征**:
```
LapTime (s): 448
LapTime_Delta: 429
RaceProgress: 372
Cumulative_Degradation: 336
TyreLife: 317
```
→ 这些特征在 train/test 分布差异最大

## 关联

- 主页: [ML Agent Memory Dashboard](../dashboard.md)
- 原则集: [16 条 Layer 3 原则](../16-principles.md) → adversarial_validation_limitation
- 经验证据: [s6e5 — F1 停车预测（净化不改善）](../competitions/s6e5.md)

---

更新: 2026-05-24 | 来源: adversarial-validation-implementation skill + v16 实验
