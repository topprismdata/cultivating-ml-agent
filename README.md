# Cultivating ML Agent Expert

> A systematic guide for training AI agents from ML novice to competition Top 10% through knowledge crystallization.

Based on 2 months of real-world experimentation with Claude Code, covering 13 ML competitions/projects across Re-ID, time series forecasting, tabular data, quantitative Alpha, medical imaging, audio classification, and mathematical reasoning.

**Key Result**: Agent went from needing 2 weeks to achieve Top 10% in its first competition to achieving Top 5% — a **14x speedup** attributable entirely to accumulated and crystallized knowledge.

## Quick Start

### For Humans

1. Read the [main guide](docs/cultivating-ml-agent-expert.md) (1088 lines, ~30 min)
2. Browse [example skills](skills/examples/) for concrete patterns
3. Use [templates](templates/) to create your own skills

### For AI Agents (Claude Code, etc.)

1. Read [AGENTS.md](AGENTS.md) for autonomous ML workflow instructions
2. Use `framework/` modules for structured pipeline (config, logging, validation, MLflow)
3. Activate skills from `skills/examples/` when encountering matching problems

## Core Concepts

### 1. Nurture-First Development

Don't pre-program all knowledge. Instead, build a **Knowledge Crystallization Cycle**:
encounter problems in practice → extract reusable patterns → organize into structured skills → auto-activate on similar problems.

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

## Project Structure

```
cultivating-ml-agent/
├── README.md                    # This file
├── AGENTS.md                    # Autonomous agent instructions
├── docs/
│   ├── cultivating-ml-agent-expert.md   # Main guide (1088 lines)
│   └── framework/                       # Framework documentation
│       ├── config-spec.md               # Configuration specification
│       ├── experiment-workflow.md        # Experiment workflow guide
│       ├── mlops-pipeline.md            # MLOps pipeline design
│       └── project-template.md          # Project template guide
├── framework/                   # Reusable MLOps framework
│   ├── config_template.yaml     # Competition config template
│   ├── script_template.py       # Experiment script template (5-stage)
│   ├── requirements.txt         # Framework dependencies
│   └── src/                     # Python modules
│       ├── config.py            # CompetitionConfig dataclass
│       ├── pipeline/
│       │   ├── mlflow_utils.py  # MLflow experiment tracking
│       │   └── validate.py      # Data validation + evaluation gates
│       └── utils/
│           ├── logging_utils.py # ExperimentLogger
│           ├── metrics.py       # Competition metrics
│           ├── paths.py         # Directory resolution
│           └── submission.py    # Submission validation
├── skills/
│   └── examples/                # 19 real skills extracted from practice
│       ├── claudeception/       # Auto skill extraction system
│       ├── three-layer-wisdom-extraction/  # Wisdom abstraction
│       ├── agent-nurture-framework/       # Nurture methodology
│       └── ... (see skills/examples/ for full list)
└── templates/
    ├── bug-fix-skill.md          # Template for bug fix skills
    └── knowledge-skill.md        # Template for knowledge skills
```

## Covered Projects

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

## Key Methodologies (SOPs)

The most important content is the **5 SOPs** in the main guide:

1. **Competition Startup SOP** — Systematic workflow from data download to first submission
2. **Model Debugging SOP** — Progressive diagnosis from prediction magnitude to feature importance
3. **Skill Extraction SOP** — Automated knowledge crystallization via claudeception
4. **Experiment Management SOP** — Reproducible iteration with naming conventions
5. **Ensemble Learning SOP** — From correlation check to optimal blending

## MLOps Framework

The `framework/` directory provides reusable Python modules validated against real Kaggle competitions (H&M Recommendations LB 0.02368, S6E4 LB 0.98150).

### Quick Integration

```bash
# Copy framework to your competition project
cp -r framework/ /path/to/your-competition/

# Edit config for your competition
cp framework/config_template.yaml config.yaml
```

### Framework Components

| Component | What It Does | Validated In |
|-----------|-------------|--------------|
| `CompetitionConfig` | Type-safe YAML config with CLI overrides | R27 (H&M) |
| `ExperimentLogger` | Timestamped logging with metric formatting | R27, R18 (S6E4) |
| `validate_pipeline()` | Data quality checks at stage boundaries | R27 |
| `validate_features()` | Feature completeness + NaN/Inf detection | R27 |
| `evaluation_gate()` | Regression prevention (block bad submissions) | R27 |
| `start_experiment()` | MLflow context manager for experiment tracking | R27 |
| `validate_and_save()` | Submission format validation | R27, R18 |
| `get_submission_filename()` | Standardized naming convention | R27, R18 |

## Academic Alignment

| Academic Concept | Our Practice |
|-----------------|-------------|
| AIDE (Huang 2024) trial-and-error learning | Detailed failure records per competition |
| AutoMind (Zhang 2025) knowledge base | 120+ SKILL.md three-layer architecture |
| Voyager (Wang 2023) skill library | Claudeception auto-extraction system |
| CoMind (2025) memory architecture | Global/project/skill three-layer memory |
| Reflexion (Shinn 2023) experience reflection | Three-layer wisdom extraction |
| NFD (Zhang 2026) nurture-first | Core philosophy of this project |

## License

MIT License — feel free to use this framework to cultivate your own ML agent.

## Contributing

Contributions welcome! Especially:
- New skill examples from your own ML projects
- Improved SOPs or methodologies
- Translations of the main guide
- Bug fixes in skill templates
