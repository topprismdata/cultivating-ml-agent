# Cultivating ML Agent Expert

> A systematic guide for training AI agents from ML novice to competition Top 10% through knowledge crystallization.

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Skills](https://img.shields.io/badge/skills-28%2B-blue)](https://github.com/topprismdata/cultivating-ml-agent/tree/main/skills/examples)
[![Competitions](https://img.shields.io/badge/competitions-15%2B-success)](https://github.com/topprismdata/cultivating-ml-agent#covered-projects-15)
[![Version](https://img.shields.io/badge/version-0.7.0-orange)](https://github.com/topprismdata/cultivating-ml-agent/releases)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/topprismdata/cultivating-ml-agent/pulls)
[![Stars](https://img.shields.io/github/stars/topprismdata/cultivating-ml-agent?style=social)](https://github.com/topprismdata/cultivating-ml-agent/stargazers)

</div>

[中文版本](README.md) | Chinese Version

---

Based on **3+ months of real-world experimentation** with Claude Code, covering **15+ ML competitions/projects** across Re-ID, time series forecasting, tabular data, quantitative Alpha, medical imaging, audio classification, and mathematical reasoning.

**Key Result**: Agent went from needing 2 weeks to achieve Top 10% in its first competition to achieving Top 5% — a **14x speedup** attributable entirely to accumulated and crystallized knowledge.

---

## 🆕 What's New (v0.7.0, 2026-06-14) — AutoGluon Era

### 🏆 Latest Achievements

| Competition | Best Score | Method |
|-------------|------------|--------|
| **House Prices Advanced Regression** | **0.11750 LB** | User blended V16+V17 (30/70) — beat cs229_v9 0.11765 |
| **Spaceship Titanic** | **0.80780 LB** | SST_v2 Top-5 CatBoost — beat V12 0.8066 |
| **AutoGluon Historical Re-run** | 4/12 wins | AutoGluon beats manual 2/4 times |

### 📚 New Skills (28+ total, 3 new in v0.7.0)

| Skill | Purpose | Validated |
|-------|---------|-----------|
| **autogluon-first** | Run AutoGluon `best_quality` as first step in any tabular competition (5-15 min baseline) | HP CV 0.1180 vs V18 0.1194 |
| **catboost-first-tabular** | When manual GBDT needed, start with CatBoost (5-variant ensemble sweet spot) | SST OOF 0.8124 vs XGB 0.8003 |
| **cv-lb-gap-acknowledgment** | CV improvement does NOT equal LB improvement. Always validate on LB | SST 0.005-0.01 gap |

### ✏️ Enhanced Skills (2 in v0.7.0)
- **ml-sweet-spot** — Added CatBoost-First evidence + AutoGluon-First comparison
- **kaggle-optimal-blending** — Added asymmetric-blending (30/70) principle

---

## 🚀 Quick Start

### 👤 For Humans

1. Read the [main guide](docs/cultivating-ml-agent-expert.md) (1088 lines, ~30 min)
2. Browse [example skills](skills/examples/) — **28+ skills** covering tabular, NLP, vision, time series
3. Use [templates](templates/) to create your own skills
4. **🆕 v0.7.0**: Start with `skills/examples/autogluon-first/` for tabular problems

### 🤖 For AI Agents (Claude Code, etc.)

1. Read [AGENTS.md](AGENTS.md) for autonomous ML workflow instructions
2. Use `framework/` modules for structured pipeline (config, logging, validation, MLflow)
3. Activate skills from `skills/examples/` when encountering matching problems
4. **🆕 v0.7.0**: Try [`ml-agent-code-template/`](ml-agent-code-template/) for a ready-to-use Claude Code setup with auto-activation, cross-model review (Antigravity/Gemini/Codex), and memory health checks.

---

## 💡 Core Concepts

### 1. Nurture-First Development

Don't pre-program all knowledge. Instead, build a **Knowledge Crystallization Cycle**:
> Encounter problems in practice → extract reusable patterns → organize into structured skills → auto-activate on similar problems.

### 2. Three-Layer Knowledge Architecture

| Layer | Content | Update Frequency |
|-------|---------|-----------------|
| L1: Core Capabilities | ML fundamentals, data science workflow | Monthly |
| L2: Domain Skills | Competition-specific techniques, anti-patterns | Weekly |
| L3: Wisdom Principles | Cross-domain universal principles | Per milestone |

### 3. Five-Stage Learning Loop

```
Study (Theory) → Verify (Notebook) → Apply (Competition) → Extract (Crystallize) → Plan (Gap Analysis)
```

### 4. 🆕 v0.7.0: AutoML-First Strategy

For any **tabular** problem, **AutoGluon is the optimal first step** (5-15 min). Only move to manual GBDT work if:
- AutoGluon's OOF doesn't meet requirements
- You have domain knowledge AutoGluon can't capture
- You want to add AutoGluon as a "Silver" signal in a custom pipeline

**Validated**: AutoGluon beats manual ensembles **2/4 times** on small/medium tabular datasets.

---

## 📁 Project Structure

```
cultivating-ml-agent/
├── README.md                    # This file (English)
├── README_EN.md                 # Chinese version
├── AGENTS.md                    # Autonomous agent instructions
├── docs/
│   ├── cultivating-ml-agent-expert.md   # Main guide (1088 lines)
│   └── framework/                       # Framework documentation
├── framework/                   # Reusable MLOps framework
├── skills/
│   └── examples/                # 28+ real skills extracted from practice
│       ├── autogluon-first/            # 🆕 v0.7.0
│       ├── catboost-first-tabular/     # 🆕 v0.7.0
│       ├── cv-lb-gap-acknowledgment/   # 🆕 v0.7.0
│       ├── claudeception/              # Auto skill extraction
│       ├── three-layer-wisdom-extraction/
│       ├── agent-nurture-framework/
│       ├── ml-sweet-spot/              # ✏️ v0.7.0
│       ├── kaggle-optimal-blending/     # ✏️ v0.7.0
│       └── ... (28+ total)
└── templates/
    ├── bug-fix-skill.md
    └── knowledge-skill.md
```

---

## 🏆 Covered Projects (15+)

| # | Project | Domain | Key Achievement |
|---|---------|--------|----------------|
| 1 | Kaggle S6E2 | Tabular | First competition, Top 9% |
| 2 | Kaggle S6E3 | Tabular | Adversarial validation breakthrough |
| 3 | Kaggle S6E4 | Spatiotemporal Graph | 24h to Top 10% |
| 4 | WorldQuant Brain Alpha | Quantitative | Alpha factor mining |
| 5 | Jaguar Re-ID | Computer Vision | 94.46% validation accuracy |
| 6 | AIMO3 | Math Reasoning | SC-TIR with Qwen3.5 |
| 7 | Store Sales | Time Series | LB 1.859 → 0.399 (4.7x improvement) |
| 8 | Vesuvius Challenge | 3D Segmentation | nnU-Net + RAG research |
| 9 | BirdCLEF+ 2026 | Audio Classification | 234 wildlife species |
| 10 | March Madness 2026 | Sports Prediction | Elo/Massey rating system |
| 11 | ISEC 2026 | Software Defects | SMOTE + polynomial features |
| 12 | Store Sales R11 | Time Series | Top 5% (latest) |
| 13 | nnU-Net Medical | Medical Imaging | Apple Silicon training |
| 14 | **House Prices Advanced Regression** | **Tabular** | **🆕 v0.7.0: LB 0.11750** |
| 15 | **Spaceship Titanic** | **Tabular** | **🆕 v0.7.0: LB 0.80780** |

---

## 🛠️ Key Methodologies (SOPs)

The most important content is the **5 SOPs** in the main guide:

1. **Competition Startup SOP** — Systematic workflow from data download to first submission
2. **Model Debugging SOP** — Progressive diagnosis from prediction magnitude to feature importance
3. **Skill Extraction SOP** — Automated knowledge crystallization via claudeception
4. **Experiment Management SOP** — Reproducible iteration with naming conventions
5. **Ensemble Learning SOP** — From correlation check to optimal blending

### 🆕 v0.7.0 New Methodological Insights

| Insight | Why It Matters |
|---------|----------------|
| **AutoGluon is the first step** for tabular | 5-15 min baseline matches days of manual work |
| **CV ≠ LB** | CV improvement does NOT translate to LB (0.005-0.01 gap common) |
| **CatBoost > LightGBM/XGBoost** on tabular | Native categorical handling, robust defaults |
| **Multi-model diversity > multi-seed** | 3 GBDT families > 15 same-family models |
| **Asymmetric blending** | 30% Silver + 70% Top-5 > 50/50 (when one family dominates) |

---

## 🔧 MLOps Framework

The `framework/` directory provides reusable Python modules validated against real Kaggle competitions (H&M Recommendations LB 0.02368, S6E4 LB 0.98150).

### Quick Integration

```bash
# Copy framework to your competition project
cp -r framework/ /path/to/your-competition/

# Edit config for your competition
cp framework/config_template.yaml config.yaml
```

### 🆕 v0.7.0: Recommended Workflow for Tabular

```
Step 1: AutoGluon (5-15 min)      [NEW SKILL: autogluon-first]
   ↓ Validate OOF
Step 2: CatBoost single model     [NEW SKILL: catboost-first-tabular]
   ↓ Compare
Step 3: 5 CatBoost variants ensemble (sweet spot)
   ↓ Add LightGBM + XGBoost (multi-model diversity)
Step 4: Validate on LB              [NEW SKILL: cv-lb-gap-acknowledgment]
   ↓ Stop if no improvement
Step 5: AutoGluon as Silver + Custom ensemble  [NEW SKILL: kaggle-optimal-blending]
   ↓ Submit
```

---

## 🎓 Academic Alignment

| Academic Concept | Our Practice |
|-----------------|-------------|
| AIDE (Huang 2024) trial-and-error learning | Detailed failure records per competition |
| AutoMind (Zhang 2025) knowledge base | 120+ SKILL.md three-layer architecture |
| Voyager (Wang 2023) skill library | Claudeception auto-extraction system |
| CoMind (2025) memory architecture | Global/project/skill three-layer memory |
| Reflexion (Shinn 2023) experience reflection | Three-layer wisdom extraction |
| NFD (Zhang 2026) nurture-first | Core philosophy of this project |
| **AutoGluon (Fakoor 2020)** | **🆕 v0.7.0: Multi-algorithm ensemble + stacking baseline** |
| **TabPFN (Hollmann 2023)** | **🆕 v0.7.0: Transformer for small tabular (future direction)** |

---

## 📜 Changelog

### v0.7.0 (2026-06-14) — AutoGluon Era

**Added 3 new skills:**
- `autogluon-first` — Run AutoGluon `best_quality` as first step
- `catboost-first-tabular` — CatBoost > LightGBM/XGBoost for tabular
- `cv-lb-gap-acknowledgment` — CV improvement ≠ LB improvement

**Enhanced 2 skills:**
- `ml-sweet-spot` — Added CatBoost-First evidence
- `kaggle-optimal-blending` — Added asymmetric-blending (30/70)

**New competitions covered:**
- House Prices Advanced Regression (LB 0.11750)
- Spaceship Titanic (LB 0.80780)

**Validation:**
- AutoGluon beats manual ensembles 2/4 times on small/medium tabular
- Top-5 CatBoost ensemble > 15-model mixed ensemble (when one family dominates)
- 0.005-0.01 CV-LB gap consistently observed

### v0.6.0 (2026-06-02) — Proactive Evolution

- Added 3 Proactive Evolution enhancements
- Updated nurture framework
- Added retail-eda-framework skill

### v0.5.0 (2026-05-31) — ML Agent Code Template

- Added `ml-agent-code-template/` (9 hooks, 6 commands, 2 agents)
- Validated on 8 MLE-Bench competitions (6 Gold, 2 Silver)
- Added Obsidian Memory Vault pattern

### Earlier (v0.1.0 - v0.4.0)

- 13 competition experiences crystallized
- 19 → 28+ skills added
- Three-layer knowledge architecture established

---

## 📄 License

MIT License — feel free to use this framework to cultivate your own ML agent.

## 🤝 Contributing

Contributions welcome! Especially:
- **New skill examples from your own ML projects**
- **Improved SOPs or methodologies**
- **Translations of the main guide**
- **Bug fixes in skill templates**
- **New AutoML tool integrations** (H2O, FLAML, Auto-sklearn)

### Adding a New Skill

1. Copy `templates/knowledge-skill.md` to `skills/examples/<your-skill-name>/SKILL.md`
2. Fill in the template (problem, context, solution, anti-patterns)
3. Validate on at least one real competition
4. Update the index in this README
5. Submit a PR with the new skill

---

<div align="center">

**Last Updated**: 2026-06-14 | **Version**: 0.7.0 | **Total Skills**: 28+ | **Total Competitions**: 15+

Made with ❤️ for the ML community

</div>
