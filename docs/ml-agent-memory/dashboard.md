# ML Agent Memory Dashboard

> 可被所有 agent 通过文件系统直接读取的经验知识库

## 结构

```
~/obsidian/ml-agent-memory/
├── dashboard.md          # 本文件 — 快速查询入口
├── competitions/         # 7 个竞赛的结构化记录
├── experiments/          # 28 条实验记录（时间线）
├── principles/           # 16 条 Layer 3 通用原则
└── skills/              # 5 项核心技能的决策框架
```

## 快速查询

### 按竞赛查实验
- `competitions/s6e5.md` — F1 停车预测，最佳 LB 0.9526
- `competitions/s6e4.md` — 灌溉预测，最佳 LB 0.98150（外部预测主导）

### 按决策点查原则
- **特征工程策略** → `principles/16-principles.md` → ground_truth_encoding, complexity_budget
- **CV 策略** → `principles/16-principles.md` → distribution_mismatch, adversarial_validation_limitation
- **集成方法** → `principles/16-principles.md` → quality_over_quantity, consensus_anchor
- **外部数据融合** → `principles/16-principles.md` → work_smart_not_hard, leverage_principle

### 按诊断信号查
- CV-LB gap > 0.01 → `skills/adversarial-validation.md` → AUC≈0.50 → 停止净化
- lag feature 全 NaN → `principles/16-principles.md` → `preds.mean()/train.mean()`
- 模型相关性 > 0.97 → `principles/16-principles.md` → signal_dilution

## 最新发现（v16 实验）

| 发现 | 结论 |
|------|------|
| Adversarial AUC ≈ 0.50 → train/test 已对齐 | 净化过滤徒劳，立即转向其他方向 |
| v16 净化后 60K ≈ 完整 439K（CV 0.95290）| CV-LB gap ~0.01 非分布不匹配造成 |
| S6E5 最佳 CV 0.9622，LB 0.9512 | 需探索 GroupKFold 或领域特征时机 |

## Layer 3 核心原则（5 条最高价值）

1. **work_smart_not_hard**: 外部预测整合 ROI ~7x 自训练
2. **local_optimum_trap**: 3 次 <0.0001 改进 → 必须 pivot
3. **simple_diagnostic**: 1 行诊断 > 复杂调试
4. **quality_over_quantity**: 4-6 个高质量源 > 23 个杂源
5. **adversarial_validation_limitation**: AUC≈0.50 时停止净化

## Agent 查询示例

```python
# 在任何 agent 的上下文中，直接读取：
with open("~/obsidian/ml-agent-memory/competitions/s6e5.md") as f:
    insights = f.read()
```

---

更新: 2026-05-24 | 实验数: 28 | 竞赛数: 7 | 原则数: 16
