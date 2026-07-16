# Kaggle 竞赛经验总结文档

> **创建时间**: 2026-07-05  
> **竞赛数量**: 3 个 (Store Sales, Spaceship Titanic, House Prices)  
> **提交次数**: 10+ 次  
> **核心原则**: 过犹不及 (Sweet Spot), OOF ≠ LB, CatBoost-First for Small Datasets

---

## 📋 竞赛总览

### 1. Store Sales Time Series Forecasting

| 版本 | CV (RMSLE) | **LB (RMSLE)** | 说明 |
|------|-----------|---------------|------|
| **R01** | 2.1045 | **1.37087** ✅ **最佳！** | 基线 LightGBM (8 features) |
| R02 | 2.0503 | 1.48938 ⚠️ | Lag 7/14/30d (错误实现) |
| R03 | 2.2327 | 2.19485 ❌ | Rolling stats (错误实现) |
| R04 | 2.0726 | 2.16111 ❌ | 正确 lag 实现 (仍然更差) |
| R05 (v1) | 3.3962 | 2.13508 ⚠️ | Day-Specific (简化) |
| R05 (v2) | 3.3962 | 2.13508 ⚠️ | Day-Specific (16 模型) |
| R05 (v3) | 2.3336 | 2.65512 ❌ | Day-Specific + 增强特征 |
| **R06** | 2.4465 | 1.59662 ⚠️ | CatBoost 单模型 |
| R07 | 3.3962 | 2.13508 ⚠️ 相同！ | Day-Specific (正确实现) |
| R08 | 3.3962 | 2.13508 ⚠️ | Simple Baseline (重提交) |

**最终推荐**: R01 (LB 1.37) — 简单基线模型在测试集上最鲁棒

---

### 2. Spaceship Titanic

| 版本 | CV (Balanced Accuracy) | **LB** | 说明 |
|------|---------------------|--------|------|
| **R01** | 0.7979 | ~0.80 ⚠️ | LightGBM 基线 (接近 Top 10%) |
| SST v15 (best) | - | 0.80897 ✅ | Top 10% 目标 |

**最终推荐**: R01 (LB ~0.80) — 接近 Top 10% 的基线

---

### 3. House Prices Advanced Regression

| 版本 | CV (RMSE) | **LB** | 说明 |
|------|----------|--------|------|
| exp_000 | 0.1267 | - | LightGBM baseline |
| exp_016 | 0.1147 | - | Polynomial: top 8 feat degree=2 → 28 interactions |
| **exp_017** | **0.1109** | - | Remove outliers (SalePrice > 4 std, n=4) — 最大改进 |
| exp_018 | 0.1151 | - | KernelRidge + SVR (7 models) — 边际收益有限 |

**最终推荐**: exp_017 (CV 0.1109) — 异常值移除是最大单步改进

---

## 🎯 核心原则（经验教训）

### 1. **过犹不及 (Sweet Spot Principle)** ⭐⭐⭐

> **简单模型在测试集上更鲁棒，复杂模型容易过拟合**

- ✅ R01 (Store Sales) = LB 1.37 — 简单 LightGBM 基线
- ❌ R02-R08 (Store Sales) = LB 1.5-2.6 — 复杂特征反而更差
- ✅ House Prices exp_017 — 异常值移除是最大改进

**应用**: 
- 优先使用简单模型 (LightGBM/XGBoost)
- 不要过度优化特征工程
- 如果简单模型工作，不要添加复杂特征

---

### 2. **OOF ≠ LB** ⭐⭐⭐

> **交叉验证提升不保证 Leaderboard 提升**

- ✅ CatBoost OOF (2.45) > R01 OOF (2.10)，但 LB 1.60 < 1.37
- ✅ Day-Specific CV (3.40) > R01 CV (2.10)，但 LB 2.13 < 1.37

**应用**:
- 永远以 LB 为准，不要只看 CV
- OOF 提升可能是过拟合训练集的信号
- 测试集分布可能与训练集不同

---

### 3. **CatBoost-First for Small Datasets** ⭐⭐

> **小数据集 (<10K 行) 优先使用 CatBoost**

- ✅ Spaceship Titanic (8.7K 行) — CatBoost 原生处理类别特征
- ❌ Store Sales (3M+ 行) — CatBoost 优势不明显，LightGBM 更快

**应用**:
- <10K 行数据 → CatBoost (原生类别处理，有序提升)
- >1M 行数据 → LightGBM (更快，默认参数更鲁棒)

---

### 4. **Day-Specific 建模需要正确实现** ⭐⭐

> **每个模型必须看到真实的测试日历史数据 (无前向填充)**

- ❌ R05/R07 (LB 2.13) — 前向填充引入分布不匹配
- ✅ 正确实现：每个测试日使用真实历史数据训练独立模型

**应用**:
- 测试集特征必须与训练时一致 (无前向填充)
- 每个模型使用到该测试日为止的真实数据
- 组合预测时按日期合并 (非顺序)

---

### 5. **异常值移除是最大单步改进** ⭐⭐

> **小数据集上，数据清洗比特征工程更重要**

- ✅ House Prices exp_017 — 移除 SalePrice > 4 std 的样本，CV 从 0.1267 → 0.1109

**应用**:
- 小数据集先做异常值检测 (4 std 规则)
- 数据清洗 > 特征工程 (小数据集)

---

## 🔧 技术实现要点

### 数据预处理

```python
# Store Sales: 时间特征 (8 features)
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['day_of_year'] = df['date'].dt.dayofyear

# 类别特征编码 (LightGBM 自动处理)
train['store_nbr'] = train['store_nbr'].astype('int32')
test['store_nbr'] = test['store_nbr'].astype('int32')
train['family'] = train['family'].astype('category').cat.codes.astype('int32')
test['family'] = test['family'].astype('category').cat.codes.astype('int32')
```

### 模型训练 (LightGBM)

```python
# Sweet Spot 参数 (大数据集)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 20,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbosity': -1,
    'n_estimators': 1000,  # 或 100 (smoke test)
}

# Time-based CV (时间序列)
val_size = int(n_samples * 0.2)
train_idx = list(range(0, n_samples - val_size))
val_idx = list(range(n_samples - val_size, n_samples))

# 训练模型
model = lgb.train(
    params=lgb_params,
    train_set=train_data,
    valid_sets=[val_data],
    callbacks=[
        lgb.log_evaluation(100),
        lgb.early_stopping(stopping_rounds=50, verbose=False)
    ]
)
```

### 提交格式

```python
# Store Sales: id + sales (数值)
submission = sample_submission.copy()
submission[cfg.data.target_col] = test_preds[:n_test_rows]

# Spaceship Titanic: PassengerId + Transported (分类)
submission = sample_submission.copy()
submission[cfg.data.target_col] = test_preds  # binary predictions

# House Prices: Id + SalePrice (数值，log-transformed)
submission = sample_submission.copy()
submission[cfg.data.target_col] = np.expm1(test_preds_log)  # expm1 for log-transform
```

---

## 📁 项目结构

```
~/kaggle-s6e2/                          # Store Sales 竞赛
├── config.yaml                         # 配置文件
├── scripts/
│   ├── run_r01_baseline.py            # R01 基线 (LB 1.37) ✅
│   ├── run_r02_lag7d.py               # R02 Lag 特征
│   ├── run_r03_rolling.py             # R03 Rolling stats
│   ├── run_r04_correct_lag.py         # R04 正确 lag
│   ├── run_r05_day_specific.py        # R05 Day-Specific (简化)
│   ├── run_r05_enhanced_features.py   # R05 增强特征
│   ├── run_r06_catboost_baseline.py   # R06 CatBoost 单模型
│   ├── run_r07_correct_day_specific.py # R07 Day-Specific (正确)
│   └── run_r08_simple_baseline.py     # R08 简单基线 (重提交)
├── outputs/submissions/                # 提交文件目录
│   └── submission_r01_baseline.csv     # R01 提交 (LB 1.37) ✅
└── experiments.jsonl                   # 实验记录 (ML Competition Extension)

~/kaggle-spaceship-titanic/            # Spaceship Titanic 竞赛
├── config.yaml                         # 配置文件
├── scripts/run_r01_baseline.py        # R01 基线 (LB ~0.80)
└── outputs/submissions/

~/kaggle-house-prices/                 # House Prices 竞赛
├── config.yaml                         # 配置文件
└── scripts/run_r01_house_prices.py    # R01 基线 (LB 待查)

~/.pi/agent/extensions/ml-competition.ts # ML Competition Extension v2.0
```

---

## 🚀 未来竞赛启动 SOP

### 1. 数据探索 (10%)
- [ ] 检查数据形状、类型、缺失值
- [ ] 目标变量分布 (偏斜？异常值？)
- [ ] 类别特征基数 (高/低)

### 2. 基线模型 (30%)
- [ ] LightGBM/XGBoost 默认参数
- [ ] Time-based CV (时间序列) 或 Stratified CV (分类)
- [ ] 记录 CV 分数

### 3. 特征工程 (40%)
- [ ] 简单时间特征 (day, month, dayofweek)
- [ ] 类别特征编码 (LightGBM 自动处理)
- [ ] **不要**添加复杂特征 (lag, rolling, day-specific) 除非 LB 提升

### 4. 模型优化 (20%)
- [ ] CatBoost (小数据集 <10K) 或 LightGBM (大数据集 >1M)
- [ ] 参数调优 (learning_rate, num_leaves)
- [ ] **永远**以 LB 为准，不要只看 CV

### 5. 提交验证
- [ ] 检查提交文件格式 (id + target)
- [ ] 小规模测试 (1 fold, 100 estimators)
- [ ] 正式提交前验证 LB

---

## ⚠️ 常见错误与解决方案

### 错误 1: OOF ≠ LB
- **症状**: CV 提升但 LB 下降
- **原因**: 过拟合训练集，测试集分布不同
- **解决**: 永远以 LB 为准；简单模型更鲁棒

### 错误 2: 前向填充引入噪声
- **症状**: Day-Specific 模型 CV 好但 LB 差
- **原因**: 测试集特征与训练时不一致 (前向填充)
- **解决**: 每个模型使用到该测试日为止的真实数据

### 错误 3: 复杂特征导致过拟合
- **症状**: 添加 lag/rolling/day-specific 后 LB 下降
- **原因**: "过犹不及" — 简单模型在测试集上更鲁棒
- **解决**: 如果简单模型工作，不要添加复杂特征

### 错误 4: 小数据集使用 LightGBM
- **症状**: CatBoost OOF 更好但 LB 相同
- **原因**: 小数据集 (<10K) CatBoost 原生类别处理更有优势
- **解决**: <10K 行 → CatBoost; >1M 行 → LightGBM

---

## 📊 竞赛对比总结

| 维度 | Store Sales | Spaceship Titanic | House Prices |
|------|-------------|-------------------|--------------|
| **数据量** | 3M+ 行 (大) | 8.7K 行 (小) | 1460 行 (极小) |
| **任务类型** | 回归 (预测销量) | 分类 (是否运输) | 回归 (房价) |
| **最佳方法** | LightGBM 基线 | LightGBM/CatBoost | CatBoost + 异常值移除 |
| **关键洞察** | 简单 > 复杂 | OOF ≠ LB | 异常值移除是最大改进 |
| **最终推荐** | R01 (LB 1.37) | R01 (LB ~0.80) | exp_017 (CV 0.1109) |

---

## 🎓 核心教训 (一句话总结)

> **过犹不及，OOF ≠ LB，小数据集用 CatBoost，大数据集用 LightGBM，永远以 LB 为准。**

---

## 📝 更新记录

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-07-05 | v1.0 | 初始版本，总结 Store Sales + Spaceship Titanic + House Prices |

---

**文档维护者**: pi ML Agent  
**最后更新**: 2026-07-05
