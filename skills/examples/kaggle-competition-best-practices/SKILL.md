---
name: kaggle-competition-best-practices
description: |
  Kaggle竞赛最佳实践和RAG知识库管理。使用时: (1) 开始任何Kaggle竞赛项目前建立规划,
  (2) 需要查询竞赛策略、获胜方案、特征工程技巧, (3) 需要为竞赛创建NotebookLM知识库。
  涵盖完整的竞赛工作流:从数据探索、特征工程、模型训练到提交策略。包含Stage1/Stage2
  规则、获胜方案解析、ELO/Massey排名、评分函数优化等核心知识。
---

# Kaggle竞赛最佳实践与知识库管理

## Problem

Kaggle竞赛涉及大量领域知识、获胜技巧和策略。每次竞赛都重新学习效率低下。需要:
- 系统化的竞赛工作流程
- 可复制的获胜方案
- 竞赛知识的快速查询
- 为每个竞赛建立RAG知识库

## Context / Trigger Conditions

使用此技能当:
- **开始新的Kaggle竞赛项目** - 建立完整工作计划
- **查询竞赛策略** - 提问: "这个竞赛的最佳实践是什么?"
- **分析获胜方案** - 需要理解top方案的思路
- **特征工程困惑** - 需要有效的特征灵感
- **提交策略** - Stage1/Stage2规则和策略
- **创建竞赛知识库** - 为新竞赛整理资料

## Solution

### Phase 1: 竞赛规划与知识库建立

#### 1.1 创建竞赛NotebookLM知识库

为每个竞赛创建独立的NotebookLM notebook用于RAG查询:

```bash
# 创建新notebook
notebooklm create "{竞赛名称} 竞赛资料"

# 记录notebook ID
notebooklm list | grep "{竞赛名称}"
```

#### 1.2 收集和上传资料

爬取并整理以下内容上传:

**必须上传的资料**:
- 官方规则和数据说明
- 论坛置顶讨论
- TOP20高赞Notebooks摘要
- 获奖方案解析

**资料整理模板**:
```markdown
# {竞赛名称} - 论坛讨论

## 置顶/官方讨论
- 竞赛规则更新
- 数据发布通知
- 重要时间节点

## 热门讨论
- 按投票排序的TOP20

## 技术讨论
- 数据质量问题
- 特征工程技巧
- 模型架构

---

# {竞赛名称} - Notebooks分析

## TOP30 Notebooks
## 技术栈统计
## 获奖方案深度解析

---

# {竞赛名称} - 竞赛指南

## 任务理解
## 数据结构
## 评估指标
## 提交规则(Stage1/Stage2)
## 关键时间节点
## 最佳实践
```

**上传命令**:
```bash
notebooklm source add 论坛.md -n {notebook_id} --title "论坛讨论"
notebooklm source add notebooks.md -n {notebook_id} --title "Notebooks"
notebooklm source add guide.md -n {notebook_id} --title "竞赛指南"
```

#### 1.3 获取Notebooks列表（推荐方法）

**首选方法: Kaggle CLI** ✅

```bash
# 安装kaggle CLI（如果未安装）
pip install kaggle

# 配置API密钥
# 从 https://www.kaggle.com/settings 下载kaggle.json
# 放置在 ~/.kaggle/kaggle.json

# 获取竞赛notebooks列表（按投票排序）
kaggle kernels list --competition {competition-slug} --sort-by voteCount --page-size 50

# 示例
kaggle kernels list --competition march-machine-learning-mania-2026 --sort-by voteCount --page-size 50
kaggle kernels list --competition vesuvius-challenge-surface-detection --sort-by voteCount --page-size 50
```

**优势**:
- ✅ 无需登录
- ✅ 数据结构化（包含投票数、作者、更新时间）
- ✅ 100%可靠
- ✅ 可排序和分页

**备用方法: Playwright爬虫** ⚠️
- 仅当kaggle CLI不可用时使用
- 可能需要登录
- 参见 `discover-undocumented-web-apis` skill

#### 1.4 批量下载和上传Notebooks（完整工作流）

**问题**: 直接上传Kaggle notebooks到NotebookLM会失败
- `.ipynb` 和 `.Rmd` 文件返回 `400 Bad Request`
- NotebookLM只支持: `.md`, `.txt`, `.pdf`, `.docx`, `.xlsx` 等格式

**完整解决方案**: 列表 → 下载 → 转换 → 上传

**Step 1: 获取notebook列表**
```bash
# M5 Forecasting - TOP 20
kaggle kernels list --competition m5-forecasting-accuracy --sort-by voteCount --page-size 20

# H&M Recommendations - TOP 20
kaggle kernels list --competition h-and-m-personalized-fashion-recommendations --sort-by voteCount --page-size 20
```

**Step 2: 批量下载（需要为每个kernel创建单独目录）**
```python
#!/usr/bin/env python3
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("/tmp/forecasting_notebooks")

# M5 TOP 20 kernels
M5_KERNELS = [
    "headsortails/back-to-predict-the-future-interactive-m5-eda",
    "robikscube/m5-forecasting-starter-data-exploration",
    # ... 更多
]

def download_kernel(kernel_ref, category):
    """下载单个kernel - 关键是为每个创建单独目录"""
    kernel_name = kernel_ref.split('/')[1]
    output_path = OUTPUT_DIR / category / kernel_name  # 单独目录!
    output_path.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "kaggle", "kernels", "pull", kernel_ref,
        "-p", str(output_path), "--metadata"
    ])

for kernel in M5_KERNELS:
    download_kernel(kernel, "m5")
```

**Step 3: 转换为Markdown**
```python
#!/usr/bin/env python3
import json
from pathlib import Path

def convert_ipynb_to_md(ipynb_path):
    """Jupyter notebook → Markdown"""
    with open(ipynb_path, 'r') as f:
        nb = json.load(f)

    md_lines = [f"# {ipynb_path.stem}\n"]

    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type', 'code')
        source = ''.join(cell.get('source', []))

        if cell_type == 'markdown':
            md_lines.append(source)
        elif cell_type == 'code':
            md_lines.append(f"\n```python\n{source}\n```\n")

    return ''.join(md_lines)

def convert_rmd_to_md(rmd_path):
    """R Markdown → Markdown（已经是类MD格式）"""
    with open(rmd_path, 'r') as f:
        return f.read()

# 批量转换
for ipynb in Path("/tmp/forecasting_notebooks/m5").rglob("*.ipynb"):
    md_content = convert_ipynb_to_md(ipynb)
    output = Path("/tmp/notebooks_md") / f"m5_{ipynb.stem}.md"
    output.write_text(md_content, encoding='utf-8')
```

**Step 4: 批量上传（添加延迟避免rate limiting）**
```bash
#!/bin/bash
NOTEBOOK_ID="705ff040-f9e0-4522-941f-2389fbf24c33"

for md_file in /tmp/notebooks_md/*.md; do
    filename=$(basename "$md_file")
    title="${filename%.md}"

    notebooklm source add "$md_file" -n "$NOTEBOOK_ID" --title "$title"

    # 关键: 添加延迟避免请求过快
    sleep 3
done
```

**完整统计**:
- M5: 20 notebooks
- H&M: 20 notebooks
- 论坛讨论: 3 markdown文件
- **总计: 43 sources → NotebookLM**

**关键注意事项**:
1. ✅ 每个kernel需要单独的下载目录
2. ✅ 必须转换为markdown再上传
3. ✅ 上传需要添加3秒延迟
4. ❌ 不要直接上传.ipynb或.Rmd（会失败）

#### 1.5 爬虫失败的备用方案

**问题**: Kaggle Notebooks/Code页面可能需要登录，API拦截返回空结果

**备用方案**:

1. **使用Web Search + Web Reader获取冠军方案**:
```python
# 1. Web搜索冠军方案
搜索: "{竞赛名称} 1st place solution writeup Kaggle"

# 2. 使用Web Reader获取内容
# (通过MCP工具或直接获取)

# 3. 手动整理成markdown
```

2. **创建综合资料文件**:
```markdown
# {竞赛名称} - 冠军方案合集

## 概述
- 竞赛目标
- 任务类型
- 评估指标

## 🏆 冠军方案 (1st Place)
- 核心技术
- 模型架构
- 后处理技巧

## 🥈 亚军方案 (2nd Place)
- 特色方法
- 关键技巧

## 其他高分方案
- Top 5共性技术
- 技术栈统计

## 共同趋势
- 所有top方案的共性
- 推荐工具/库
```

3. **验证资料充分性**:
```bash
# 测试RAG是否能回答基础问题
notebooklm ask "这个竞赛的目标是什么?"
notebooklm ask "冠军方案使用了什么模型?"

# 如果回答"没有相关信息"，说明资料不足
# 需要继续补充资料
```

**注意事项**:
- ❌ 不要在资料不足时就开始RAG查询
- ✅ 至少包含: 官方规则 + 冠军方案 + 论坛讨论 + 入门指南
- ✅ 资料越多，RAG质量越高

### Phase 2: 竞赛工作流

#### 2.1 数据探索 (Day 1-2)

**必做**:
```python
# 1. 理解数据结构
import pandas as pd
df = pd.read_csv('MRegularSeasonDetailedResults.csv')
print(df.head())
print(df.info())

# 2. 基础统计
print(f"Seasons: {df['Season'].unique()}")
print(f"Games: {len(df)}")

# 3. 目标变量分布
tourney = pd.read_csv('MNCAATourneyDetailedResults.csv')
tourney['score_diff'] = tourney['WScore'] - tourney['LScore']
print(f"Mean score diff: {tourney['score_diff'].mean():.2f}")
```

**查询知识库**:
```bash
notebooklm ask "{竞赛}数据文件有什么注意事项?"
notebooklm ask "常见的特征工程技巧有哪些?"
```

#### 2.2 特征工程 (Day 2-4)

**核心特征框架**:

```python
# A. 基础统计特征
def calculate_team_stats(games_df):
    stats = games_df.groupby(['Season', 'TeamID']).agg({
        'WScore': ['mean', 'std'],  # 得分
        'LScore': ['mean', 'std'],  # 失分
        'WFGM': ['mean', 'sum'],    # 投篮
        'WFGA': ['mean', 'sum'],    # 投篮次数
        'WFTM': ['mean', 'sum'],    # 罚球
        'WFTA': ['mean', 'sum'],    # 罚球次数
    }).reset_index()

    stats['win_rate'] = wins / (wins + losses)
    stats['avg_point_diff'] = stats['WScore_mean'] - stats['LScore_mean']
    return stats

# B. ELO评分 (使用已有实现)
from kaggle_scraper import calculate_elo

# C. Massey序数排名
massey = pd.read_csv('MMasseyOrdinals.csv')
massey_features = massey.groupby(['Season', 'TeamID'])['OrdinalRank'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max'),
    ('latest', 'last')
])

# D. 四因子分析
def calculate_four_factors(stats_df):
    # eFG% = (FGM + 0.5 * FGM3) / FGA
    stats['eFG_pct'] = (stats['WFGM'] + 0.5 * stats['WFGM3']) / stats['WFGA']
    # TOV% = TO / (FGA + TO + FTA) approximately
    # ORB% = OR / (OR + DR)
    stats['tov_pct'] = stats['WTO'] / (stats['WFGA'] + stats['WTO'])
    stats['orb_pct'] = stats['WOR'] / (stats['WOR'] + stats['WDR'])
    return stats
```

**查询知识库**:
```bash
notebooklm ask "如何计算ELO评分?"
notebooklm ask "Massey排名如何使用?"
notebooklm ask "四因子分析是什么?"
```

#### 2.3 模型训练 (Day 3-5)

**推荐模型栈**:
```python
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X, seasons):
    # 确保验证集赛季在训练集之后
    assert seasons[train_idx].max() < seasons[val_idx].min()
```

**模型集成**:
```python
models = {
    'xgb': XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.01),
    'lgbm': LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.01),
    'cat': CatBoostRegressor(iterations=500, depth=6),
    'histgb': HistGradientBoostingRegressor(max_iter=500)
}

# 加权集成
preds = [m.fit(X_train, y_train).predict(X_test) for m in models.values()]
final_pred = np.average(preds, axis=0, weights=[0.3, 0.3, 0.2, 0.2])
```

**查询知识库**:
```bash
notebooklm ask "最优的模型集成策略是什么?"
notebooklm ask "如何设置超参数?"
```

#### 2.4 概率校准 (Day 4-5)

**关键: 选择正确的评分函数**

**Brier Score vs Log Loss**:
- **Log Loss**: `log_loss(y_true, y_pred)` - 标准分类指标
- **Brier Score**: `mean((y_true - y_pred)^2)` - 概率校准指标
- **半球形评分**: 获奖方案推荐,更符合实际LB评分

```python
from scipy.stats import norm

def point_diff_to_brier(diff, std_dev=11):
    """分差转Brier Score优化概率"""
    # 使用正态CDF
    prob = norm.cdf(diff / std_dev)
    return np.clip(prob, 0, 1)

# Goto Conversion (获奖者方法)
def goto_conversion(prob, std_dev=11):
    """获奖者的概率校准方法"""
    # 使用标准误调整
    adjusted_prob = prob * std_dev
    return np.clip(adjusted_prob, 0, 1)
```

**查询知识库**:
```bash
notebooklm ask "什么是半球形评分?"
notebooklm ask "如何校准概率?"
notebooklm ask "Brier Score和Log Loss有什么区别?"
```

#### 2.5 Stage1/Stage2策略

**Stage1** (验证阶段):
- 可多次提交,用于验证模型
- 尝试不同特征组合
- 调整超参数
- **目标**: 寻找最优模型配置

**Stage2** (最终提交):
- **只允许一个提交!** ⚠️
- 选择最稳定的模型
- 不能使用对冲策略
- **截止时间前必须提交**

**查询知识库**:
```bash
notebooklm ask "Stage1和Stage2有什么区别?"
notebooklm ask "如何在Stage2选择最佳模型?"
```

### Phase 3: 常见问题与解决方案

#### 3.1 数据泄露检测

**症状**: CV分数极好但LB很差

**检查方法**:
```python
# 检查时间顺序
if 'DayNum' in df.columns:
    df['is_leak'] = df['DayNum_X'] < df['DayNum_Y']
    print(f"Leak samples: {df['is_leak'].sum()}")

# 检查未来信息
leak_features = [c for c in X.columns if '2026' in c and 'rank' not in c]
print(f"Potential leak features: {leak_features}")
```

#### 3.2 概率分布问题

**症状**: 预测集中在0.5附近

**解决**:
```python
# 检查分布
import matplotlib.pyplot as plt
plt.hist(predictions, bins=50)
plt.show()

# 解决方案
# 1. 增加特征多样性
# 2. 使用概率校准
# 3. 调整模型阈值
```

#### 3.3 过拟合LB

**症状**: LB排名不稳定

**解决**:
```python
# 使用更简单的模型
# 增加正则化
# 时间序列CV验证
# 早停策略
```

### Phase 4: 提交与迭代

#### 4.1 提交前检查清单

```python
def validate_submission(submission_df):
    """验证提交文件"""
    checks = {
        '行数': len(submission_df) == EXPECTED_ROWS,
        'ID格式': submission_df['ID'].str.match(r'^\d+_\d+_\d+$').all(),
        'Pred范围': submission_df['Pred'].between(0, 1).all(),
        '无NaN': submission_df['Pred'].notna().all(),
        '非零方差': submission_df['Pred'].std() > 0.001,
    }

    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}: {result}")

    return all(checks.values())
```

#### 4.2 RAG使用的正确心态

**关键原则**: RAG是增强能力，而非替代思考

**正确的工作流程**:
```
1. 独立思考 → 分析问题，制定初步方案
2. RAG验证补充 → 检查思路，查漏补缺
3. 两者结合 → 根据RAG信息优化方案
```

**❌ 错误做法**:
- 完全依赖RAG回答，不做独立分析
- 资料不足就开始RAG查询（会返回"没有相关信息"）
- 把RAG当Google搜索用（应该问具体技术问题）

**✅ 正确做法**:
- 先自己分析问题，记录疑问点
- 用RAG验证理解和获取最佳实践
- 结合RAG信息和自己的判断做决策

**RAG擅长的问题类型**:
- ✅ "这个竞赛的评估指标是什么？"
- ✅ "冠军方案的核心技术是什么？"
- ✅ "新手常见的错误有哪些？"
- ❌ "帮我写这个代码"（太具体）
- ❌ "这个bug怎么修"（需要调试信息）

#### 4.3 RAG查询示例

**开始竞赛时**:
```bash
notebooklm ask "{竞赛名称} 竞赛的基本规则是什么?"
notebooklm ask "评估指标是什么?如何优化?"
notebooklm ask "常见的特征工程陷阱有哪些?"
```

**特征工程阶段**:
```bash
notebooklm ask "如何处理时间序列数据?"
notebooklm ask "什么是Massey排名?如何使用?"
notebooklm ask "ELO评分系统如何实现?"
```

**模型训练阶段**:
```bash
notebooklm ask "最优的模型架构是什么?"
notebooklm ask "如何设置超参数?"
notebooklm ask "模型集成如何加权?"
```

**提交前**:
```bash
notebooklm ask "Stage2提交有什么注意事项?"
notebooklm ask "如何校准概率?"
notebooklm ask "常见的提交错误有哪些?"
```

## Verification

**知识库创建验证**:
```bash
# 1. 检查sources是否ready
notebooklm source list -n {notebook_id}

# 2. 测试RAG查询
notebooklm ask "测试问题"
```

**工作流验证**:
1. ✅ 能成功创建notebook
2. ✅ 能上传markdown文件
3. ✅ RAG能返回相关答案
4. ✅ 文件内容被正确索引

## Example

**完整竞赛流程**:

```bash
# 1. 创建知识库
notebooklm create "Titanic - Machine Learning from Disaster"

# 2. 上传资料
notebooklm source add forum.md -n abc123 --title "论坛讨论"
notebooklm source add notebooks.md -n abc123 --title "Notebooks"
notebooklm source add guide.md -n abc123 --title "竞赛指南"

# 3. 竞赛过程中查询
notebooklm ask "这个竞赛的数据有什么特点?"
notebooklm ask "特征工程有什么技巧?"
notebooklm ask "获胜方案的核心思路是什么?"
```

## Notes

### 关键注意事项

1. **数据泄露**: #1 竞赛杀手
   - 检查时间顺序
   - 不要使用未来数据
   - 验证集必须晚于训练集

2. **Stage2单一提交**: 2026年新规则
   - 不能对冲
   - 必须选择最稳定的模型
   - 手动调整需谨慎

3. **评分函数选择**:
   - Log Loss: 标准但可能不等于LB
   - Brier Score: 概率校准
   - 球形评分: 获奖方案推荐

4. **概率校准**:
   - 原始概率可能不校准
   - 使用标准差调整
   - 考虑长尾热门偏差

### 竞赛类型差异

| 竞赛类型 | 技术栈 | 注意事项 | 典型竞赛 |
|---------|--------|---------|---------|
| 表格预测 | XGBoost, LightGBM, CatBoost | 数据泄露风险高, 时间序列CV | March ML Mania, M5 Forecasting |
| 3D图像分割 | nnU-Net, 3D U-Net, 后处理 | 拓扑正确性, 孔洞填充, 连续性 | Vesuvius Challenge |
| 2D图像分类 | CNN, ResNet, EfficientNet | 计算资源限制, 数据增强 | CIFAR, ImageNet adaptations |
| NLP情感分析 | Transformer, BERT, RoBERTa | 预训练模型使用, 序列长度 | Twitter Sentiment, Jigsaw |
| 推荐系统 | 协同过滤, 深度学习, NCF | 冷启动问题, 稀疏数据 | H&M, RecSys |
| 目标检测 | YOLO, Faster R-CNN | mAP评估, anchor设计 | Global Wheat Detection |
| 时序预测 | LSTM, GRU, Temporal Fusion | 时间依赖, 特征工程 | COVID19 Forecasting |

### 竞赛时间规划

**快速竞赛 (< 1周)**:
- Day 1-2: 数据探索 + 基线模型
- Day 3-4: 特征工程 + 模型优化
- Day 5+: 集成 + 提交

**中期竞赛 (1-2周)**:
- Week 1: 深入探索 + 特征工程
- Week 2: 模型优化 + 集成

**长期竞赛 (> 1月)**:
- Month 1: 全面研究
- Month 2+: 持续优化

## References

### Kaggle资源
- [Kaggle Competitions](https://www.kaggle.com/competitions)
- [Kaggle Notebooks](https://www.kaggle.com/notebooks)
- [Kaggle Discussion Forums](https://www.kaggle.com/discussions)

### 竞赛方法论
- [Kaggle Tricks](https://www.kaggle.com/docs/competitions)
- [March Machine Learning Mania Winners](https://www.kaggle.com/c/march-machine-learning-mania-2026/discussion)
- [ELO Rating System](https://en.wikipedia.org/wiki/Elo_rating_system)
- [Brier Score](https://en.wikipedia.org/wiki/Brier_score)

### 技术论文
- "Four Factors" - Dean Oliver's basketball analytics
- "Semi-Spherical Scoring" - goto_conversation方法
- "Massey Ratings" - sports ranking systems

---

## 更新日志

**v1.3.0** (2026-03-09)
- ✨ 新增"批量下载和上传Notebooks"完整工作流 (1.4节)
  - Kaggle CLI批量下载TOP notebooks
  - 解决NotebookLM不支持.ipynb和.Rmd的问题（400 Bad Request）
  - 提供ipynb和Rmd到markdown的转换脚本
  - 批量上传脚本（包含rate limiting）
  - 验证: 39/40文件成功上传

**v1.2.0** (2026-03-08)
- ✨ 新增"获取Notebooks列表"方法 (1.3节)
  - 首选: Kaggle CLI (`kaggle kernels list`)
  - 无需登录，数据结构化，100%可靠
  - 替代Playwright爬虫（可能需要登录）
- 📝 根据实际使用经验调整方法优先级

**v1.1.0** (2026-03-08)
- ✨ 新增"爬虫失败的备用方案" (1.3节)
  - Web Search + Web Reader获取冠军方案
  - 手动创建综合资料文件
  - 资料充分性验证方法
- ✨ 新增"RAG使用的正确心态" (4.2节)
  - RAG是增强而非替代思考
  - 正确工作流程: 独立思考 → RAG验证 → 两者结合
  - RAG擅长/不擅长的问题类型
- 📊 扩展"竞赛类型差异"表格
  - 新增3D图像分割、目标检测、时序预测等类型
  - 添加技术栈列和典型竞赛列
  - 包含Vesuvius Challenge等实际案例

**v1.0.0** (2026-03-08)
- 初始版本
- 基于March ML Mania 2026经验
- 包含完整工作流程和RAG使用方法
