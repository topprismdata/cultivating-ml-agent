# Cultivating ML Agent

### A self-improving ML agent that compounds capability across projects

`Purpose: NATIVE AI` · `Maturity: APPLIED INTERNAL` · `Evidence: MULTI-PROJECT EXPERIENCE`

> Part of **TopPrism Native AI**.
>
> Competitions are used as measurable learning environments; the long-term goal is broader enterprise machine-learning work.

## Why this exists

Most ML agents are stateless in an organizational sense: a new project starts with roughly the same generic model knowledge as the previous one.

TopPrism's goal is different:

> **Every completed ML project should make the agent better at the next ML project.**

The project turns repeated practice into reusable organizational capability through knowledge crystallization, structured skills, shared tooling and explicit evaluation.

## The compounding loop

```text
Real ML Project
      ↓
Experiments & Failures
      ↓
Validated Experience
      ↓
Knowledge Crystallization
      ↓
Reusable Skill / Framework Pattern
      ↓
Automatic activation in a later project
      ↓
Better starting point for the next ML project
```

The key output is therefore not only a trained model or leaderboard score. It is a growing body of **machine-readable, reusable ML capability**.

## What compounds across projects

### Project experience

What worked, what failed, under which data/metric/compute conditions.

### Domain skills

Reusable workflows for time series, tabular ML, vision, multimodal tasks, model selection, validation, ensembling and other recurring ML problem types.

### MLOps patterns

Experiment tracking, artifact management, reproducibility, evaluation and failure recovery.

### Cross-domain principles

Higher-level lessons that survive beyond a single competition or dataset.

## Current evidence

The repository records a multi-month sequence of ML projects across multiple task families, including time series, tabular prediction, vision, medical imaging, audio, simulation/game AI, model compression and LLM-related tasks.

The strongest evidence is longitudinal rather than a single score:

- later projects reuse skills and framework components extracted from earlier projects;
- failed approaches are retained as negative knowledge;
- project-specific lessons are promoted into reusable skills only after validation;
- the shared framework and agent instructions increasingly reduce the amount of work that must be rediscovered from scratch.

### Important boundary

Competition rank is **not** the definition of success for this project. Kaggle and similar environments are useful because they provide fast, objective feedback. The intended capability is an ML agent that can improve through **any sufficiently instrumented machine-learning project**.

## Architecture

```text
                    ML Agent
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   Project Context  Skill Library   Shared Framework
        │              │              │
        └──────────────┼──────────────┘
                       ↓
                 Experiment Loop
                       ↓
             Evaluation / Evidence
                       ↓
            Knowledge Crystallization
                       ↓
             Updated Agent Capability
```

Key repository areas:

```text
AGENTS.md                 operating instructions for the agent
framework/                reusable ML / MLOps framework components
skills/                   crystallized reusable capabilities
ml-agent-code-template/   starting structure for future ML projects
templates/                reusable project artifacts
docs/                     methodology and accumulated knowledge
examples/                 worked examples
tests/                    checks for shared components
```

## Nurture-first development

The project follows a **Nurture-First** principle:

> Do not attempt to encode the complete ML playbook up front. Let real projects expose capability gaps, solve them, validate the solution, and then crystallize the reusable part.

A typical learning cycle is:

```text
Study → Verify → Apply → Extract → Plan
```

This separates speculative knowledge from knowledge that has survived practical use.

## Relationship to other TopPrism Native AI projects

### `agent-nurture-framework`

General methodology for project-driven agent capability development.

### `skill-tester`

Evaluation gate for reusable agent skills before they become trusted organizational capabilities.

### `notebook-knowledge-distillation`

Pipeline for converting external knowledge sources into validated skill candidates.

### `three-layer-wisdom-extraction`

Attempts to promote concrete project experiences into domain knowledge and cross-domain transferable principles.

## Quick start

Use the repository as a **project operating system**, not only as a code library.

1. Read `AGENTS.md`.
2. Start from `ml-agent-code-template/` for a new ML project.
3. Reuse `framework/` rather than rebuilding common MLOps plumbing.
4. Activate relevant skills from the skill library.
5. Record experiments and failures.
6. Crystallize only validated, reusable knowledge after the project.

## Evidence hygiene

Project metrics, competition scores and speed-up claims should always be tied to the specific project and protocol that produced them.

A historical observation such as a major reduction in time-to-strong-result is treated as **longitudinal case evidence**, not a universal guarantee that every future ML task will improve by the same factor.

## What this project is not

- It is not a single AutoML model.
- It is not a Kaggle-only agent.
- It is not a static prompt collection.
- It is not a claim that accumulated skills eliminate the need for project-specific reasoning.

## Roadmap

The direction of travel is toward a company-wide ML capability layer where:

- new projects start from accumulated organizational knowledge;
- project outcomes continuously improve the shared skill system;
- evaluation prevents low-quality knowledge from becoming permanent agent behavior;
- internal ML work becomes progressively more reproducible and less dependent on rediscovery by individuals.

## Where this fits at TopPrism

TopPrism uses Native AI to build a second compounding loop alongside customer Decision Intelligence:

```text
Customer projects generate experience
          ↓
Experience becomes reusable machine knowledge
          ↓
Internal agents improve
          ↓
The next customer / internal project starts stronger
```

That organizational learning loop is the primary purpose of this repository.
