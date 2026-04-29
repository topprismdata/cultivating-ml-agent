# MLOps Pipeline Design

## Overview

A 5-stage pipeline for every Kaggle competition, mapped to specific skills from the cultivating-ml-agent skill library.

```
Stage 0        Stage 1        Stage 2        Stage 3        Stage 4
项目初始化      数据探索        特征工程        模型训练        集成提交
──────────     ──────────     ──────────     ──────────     ───────
模板生成        EDA Notebook   特征构建        Baseline       Ensemble
数据下载        格式验证        编码策略        多模型对比      后处理
MLflow 初始化   分布分析        特征选择        超参调优        提交策略
config.yaml     对抗验证        边界测试        CV评估         LB记录

─────── 贯穿全流程 ───────
MLflow 追踪 │ Pipeline 验证 │ Sandbox-First │ 知识提取
```

## Stage 0: Project Initialization

**Goal**: Create the competition directory with all scaffolding.

**Steps**:
1. Create directory structure from template (see `project-template.md`)
2. Fill in `config.yaml` with competition details
3. Download data to `data/raw/`
4. Initialize MLflow experiment
5. Write initial `README.md` with competition overview

**Key files**: `config.yaml`, `README.md`

**Skills**: `kaggle-data-format-first`

## Stage 1: Data Exploration

**Goal**: Understand the data deeply before writing any features.

**Steps**:
1. Load data, run `validate_pipeline(stage="initial_load")`
2. EDA notebook: distributions, correlations, missing values
3. Identify data format (wide vs long, sparse vs dense)
4. Check for time leakage in time-series tasks
5. Optional: adversarial validation to check train/test distribution

**Key files**: `notebooks/eda.ipynb`

**Skills**: `kaggle-competition-best-practices` (RAG), `kaggle-top-solution-replication`

## Stage 2: Feature Engineering

**Goal**: Build features systematically with validation at each step.

**Steps**:
1. Start from baseline features (dates, IDs, basic aggregations)
2. Add features incrementally (one category at a time)
3. Run `validate_pipeline(stage="after_feature_engineering")` after each batch
4. Run `validate_features(df, feature_cols)` before training
5. Document features in `docs/feature_catalog.md`

**Key files**: `scripts/run_r{NN}_*.py`

**Skills**: `tabular-feature-engineering-patterns`, `kaggle-feature-boundary`

**Framework modules**: `features.encoding` (target_encode, frequency_encode, woe_encode)

## Stage 3: Model Training

**Goal**: Train and evaluate models with full MLflow tracking.

**Steps**:
1. Start with a single model baseline (LightGBM recommended)
2. Use `log_experiment()` to record all parameters and metrics
3. Compare multiple models (LGB, XGB, CatBoost)
4. Hyperparameter tuning with `lightgbm-unified-hyperparameter-tuning`
5. Record CV scores and feature importance

**Key files**: `scripts/run_r{NN}_*.py`

**Skills**: `ml-pipeline-unit-testing`, `lightgbm-unified-hyperparameter-tuning`

**Framework modules**: `pipeline.mlflow_utils`, `utils.metrics`

## Stage 4: Ensemble & Submission

**Goal**: Combine best models and submit strategically.

**Steps**:
1. Select diverse, high-quality models for ensemble
2. Use `ensemble-source-quality-over-quantity` for selection
3. Optimize blending weights
4. Validate submission with `validate_and_save()`
5. Submit via `submit_to_kaggle()`
6. Record LB score with `log_lb_score()`

**Key files**: `scripts/run_r{NN}_ensemble.py`

**Skills**: `ensemble-source-quality-over-quantity`, `kaggle-optimal-blending`

**Framework modules**: `utils.submission`

## Cross-cutting Concerns

### MLflow Tracking
Every experiment script calls `log_experiment()` with run_name, params, metrics, and feature list. After LB submission, `log_lb_score()` appends the LB result to the same run.

### Pipeline Validation
`validate_pipeline()` is called at every stage boundary to catch:
- Target leakage (target in test)
- Duplicate columns
- NaN anomalies
- Column misalignment

### Sandbox-First
All feature engineering is prototyped in isolation before being added to the pipeline. Each feature batch is validated independently.

### Knowledge Extraction
After each competition milestone, run `claudeception` to extract reusable patterns as skills. Update `three-layer-wisdom-extraction` for higher-level abstractions.
