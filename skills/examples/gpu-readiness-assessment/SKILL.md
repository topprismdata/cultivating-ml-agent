---
name: gpu-readiness-assessment
description: |
  Use when: (1) Considering switching from CPU to GPU for a Kaggle competition,
  (2) GPU quota is running low and you need to prioritize which competitions
  deserve GPU time, (3) About to start a model training run and unsure if GPU
  is worth the 30h/week quota cost, (4) Deciding between CPU AutoGluon vs GPU
  neural network. Provides a 5-gate assessment framework that produces a
  verdict: READY (GPU will help significantly), MARGINAL (GPU helps but ROI
  is low), or SKIP (CPU is sufficient). Inspired by NVIDIA's
  cupynumeric-migration-readiness, adapted for Kaggle GPU quota management.
---

# GPU Readiness Assessment

## Problem
GPU quota (30h/week) is the scarcest resource in Kaggle. It's shared across ALL
competitions. Wrong allocation = wasted hours that block other competitions.

Common mistakes:
- Using GPU for tabular competitions where GBDT on CPU is faster and better
- Running AutoGluon with GPU when CPU preset gives same result in less time
- Starting GPU training before verifying the pipeline works on CPU first
- Spending 8h GPU on feature exploration that could be done in 30 min on CPU

## The 5-Gate Assessment

Run all 5 gates in order. Stop at the first FAIL.

### Gate 1: Task Type Check
```
Is the task inherently GPU-friendly?
  ✅ Image classification/segmentation (CNN, UNet, ViT)
  ✅ NLP/LLM inference (transformer, BERT, LLaMA)
  ✅ Large-scale deep learning (>1M parameters)
  ✅ RL training (neural network policy)
  ⚠️ Time series with deep learning (LSTM/Transformer — may not beat GBDT)
  ❌ Tabular classification/regression (GBDT dominates, GPU adds nothing)
  ❌ Rule-based agent (PTCG, Orbit Wars — pure logic, no ML)
  ❌ ONNX optimization (NeuroGolf — graph manipulation, no training)

Verdict:
  ✅ → proceed to Gate 2
  ❌ → SKIP GPU, use CPU
```

### Gate 2: Data Size Check
```
Is the dataset large enough to benefit from GPU?
  >100K rows × >50 features → GPU helps for NN
  >10K images → GPU helps for CNN
  <10K rows → CPU is faster (GPU overhead > speedup)
  <1K images → CPU + transfer learning is sufficient

Kaggle-specific:
  Most Playground series (690K rows) → GBDT on CPU is optimal, GPU NOT needed
  Biohub (3D microscopy) → GPU required (UNet inference)
  ARC-AGI-3 (LLM agent) → GPU required (27B model inference)

Verdict:
  Sufficient data → proceed to Gate 3
  Insufficient data → SKIP GPU
```

### Gate 3: Model Architecture Check
```
Does the best model for this task require GPU?
  Neural network (CNN/RNN/Transformer) → GPU required for training
  Pre-trained model inference (LLM, UNet) → GPU required for reasonable speed
  GBDT (LGB/XGB/CAT) → CPU only (GPU versions exist but rarely better)
  Linear/logistic regression → CPU trivially
  Rule-based heuristic → CPU trivially

Competition evidence:
  House Prices → GBDT CPU (no GPU needed, 0.11750 LB)
  s6e7 → GBDT CPU (no GPU needed, 0.94942 LB)
  Biohub → UNet GPU (mandatory for 12h inference budget)
  PTCG → Rule-based CPU (no ML training)

Verdict:
  GPU-native model → proceed to Gate 4
  CPU-native model → SKIP GPU
```

### Gate 4: Quota ROI Check
```
Expected GPU hours vs expected LB improvement:

  Formula: expected_gain = (estimated_LB_with_GPU - current_LB)
           cost = estimated_GPU_hours / remaining_weekly_quota
           ROI = expected_gain / cost

  ROI > 0.001 per GPU-hour → PROCEED
  ROI < 0.001 per GPU-hour → MARGINAL (consider CPU alternative first)
  ROI unknown → Run 1h GPU smoke test first, measure actual improvement

Kaggle-specific ROI patterns:
  AutoGluon best_quality on tabular → 8h GPU → +0.001-0.003 → MARGINAL
  UNet training for segmentation → 4h GPU → +0.05-0.15 → PROCEED
  LLM fine-tuning → 10h GPU → +0.02-0.10 → PROCEED
  Feature engineering exploration → 0h GPU (do on CPU) → SKIP GPU
```

### Gate 5: Dependency Check
```
Are all GPU prerequisites met?
  □ CUDA-compatible GPU available (Kaggle T4/P100)
  □ Required libraries installable in no-internet Kaggle environment
  □ Pre-trained model weights available as Kaggle dataset
  □ Training/inference fits within 9-12h kernel time limit
  □ GPU quota not exhausted (check: 30h - used_this_week > estimated_hours)

If any box unchecked → BLOCKED, fix dependency before using GPU
```

## Final Verdict

| Gates Passed | Verdict | Action |
|--------------|---------|--------|
| All 5 | **READY** | Allocate GPU, proceed with training/inference |
| 4/5 | **MARGINAL** | Try CPU alternative first, use GPU only if CPU hits ceiling |
| ≤3/5 | **SKIP** | CPU is sufficient, save GPU for other competitions |

## Common GPU Waste Patterns

| Pattern | Waste | Fix |
|---------|-------|-----|
| AutoGluon on GPU for tabular | 8h quota, ~0 improvement | Use CPU preset |
| GPU for GBDT (LGB/XGB GPU mode) | 2h quota, ~0 improvement vs CPU | CPU is just as fast |
| Training NN without CPU pipeline validation first | 4h quota wasted on bugs | Validate pipeline on CPU first (1 epoch) |
| Running GPU and CPU competitions simultaneously | Quota split, both slow | Prioritize: 1 GPU comp at a time |
| Exploration/EDA on GPU | 1h quota for data preview | All EDA on CPU |

## Competition-Type Default GPU Strategy

| Competition Type | GPU Needed? | Default Strategy |
|-----------------|-------------|------------------|
| Standard Tabular | ❌ No | CPU GBDT, AutoGluon CPU preset |
| Code Competition | Sometimes | Depends on model (UNet=yes, GBDT=no) |
| Simulation | ❌ No | Rule-based or RL (but RL training may need GPU) |
| Research | ✅ Usually | Read requirements before starting |
| Playground | ❌ No | CPU is always sufficient |
| LLM Benchmark | ✅ Yes | Required for model inference |

## Evidence

- 20+ competitions assessed (Jun-Sep 2026)
- Tabular competitions (5+): CPU GBDT matched or beat GPU NN every time
- Biohub: GPU mandatory (UNet inference)
- PTCG: CPU only (rule-based agent)
- NeuroGolf: CPU only (ONNX graph manipulation)
- ARC-AGI-3: GPU mandatory (27B model, 96GB VRAM)
- GPU quota exhaustion blocked 5 competitions simultaneously (Jul 2026)

Inspired by NVIDIA's cupynumeric-migration-readiness 5-gate framework.
