---
name: llm-competition-pipeline
description: |
  Use when: (1) Entering an LLM-based Kaggle competition (open-source model
  required, prompt engineering, few-shot), (2) Need to choose between base
  models (Qwen, DeepSeek, LLaMA), (3) Designing a prompt strategy
  (zero-shot, few-shot, chain-of-thought), (4) Deciding whether fine-tuning
  is worth the GPU cost vs prompt engineering alone, (5) Setting up closed-book
  inference (no RAG, no internet). Covers the full pipeline: base model
  selection → prompt engineering → few-shot curation → evaluation →
  (optional) SFT/PEFT → submission. Inspired by NVIDIA's nemotron-customize
  pipeline structure, adapted for Kaggle LLM competition constraints.
---

# LLM Competition Pipeline

## Problem
LLM competitions (Industrial Automation, ARC-AGI-3, ai-agent-security) have
unique constraints vs traditional ML:
- **Open-source models only** (Qwen, DeepSeek, LLaMA — no GPT/Claude)
- **No internet at inference** (all weights/data must be bundled)
- **GPU time limited** (30h/week shared, 9-12h per notebook run)
- **Prompt engineering is 80% of the work** (not model architecture)
- **Closed-book setting** (no RAG, no tools, no retrieval)

## The Pipeline (6 Stages)

### Stage 1: Base Model Selection

```
Choose based on competition rules + GPU memory:

| Model | Params | VRAM (FP8) | Speed | Use Case |
|-------|--------|-------------|-------|----------|
| Qwen 2.5-7B | 7B | ~8GB | Fast | MCQ, classification |
| Qwen 3.6-27B | 27B | ~28GB | Medium | Reasoning, ARC-AGI-3 |
| DeepSeek-R1 | 7B/14B | ~8-16GB | Fast | Math, logic |
| LLaMA-3.1-8B | 8B | ~8GB | Fast | General purpose |

Decision factors:
  □ Competition rule: which model families are allowed?
  □ VRAM budget: Kaggle T4 = 16GB, P100 = 16GB, dual T4 = 30GB
  □ Inference time: 7B ~ 1-2s/query, 27B ~ 5-10s/query
  □ Task type: reasoning → larger, classification → smaller
```

### Stage 2: Prompt Engineering

```
Start simple, then add complexity:

Level 0 — Zero-shot:
  "Answer the following question: {question}\nAnswer:"

Level 1 — Instruction:
  "You are an expert in {domain}. Read the passage and answer.
   Passage: {passage}
   Question: {question}
   Provide only the letter of the correct answer."

Level 2 — Few-shot (3-5 examples):
  "Here are examples of correct answers:
   Example 1: {input} → {output}
   Example 2: {input} → {output}
   Example 3: {input} → {output}
   Now answer: {question}"

Level 3 — Chain-of-thought:
  "Think step by step, then give the final answer."

Level 4 — Dynamic few-shot (retrieve relevant examples):
  For each test question, retrieve the most similar training examples
  and include them as few-shot context.
```

**Key rule**: Each level adds latency. Level 0 is fastest. Only escalate if
OOF accuracy improves by >2%.

### Stage 3: Few-Shot Example Curation

```
Selecting the best few-shot examples:

Strategy A — Random (baseline):
  Pick 3-5 random examples from training set.

Strategy B — Class-balanced:
  Ensure examples cover all output classes equally.

Strategy C — Similarity-based (dynamic):
  For each test question, compute embedding similarity to training questions.
  Select top-K most similar as few-shot context.
  This is the most effective but adds inference latency.

Strategy D — Hard negatives:
  Include examples the model gets wrong (forces it to learn the distinction).

Validation: test each strategy on a held-out set. Pick the one with highest accuracy.
```

### Stage 4: Evaluation

```
Before submitting:

□ Local evaluation on training set (all examples)
□ Measure: accuracy, per-class F1, confusion matrix
□ Check: are failures concentrated in specific question types?
□ Check: is the model output format correct? (letter vs full text)
□ Check: latency per query × number of test queries < time limit

Common LLM evaluation traps:
  - Model outputs "The answer is C" instead of just "C" → parsing fails
  - Model refuses to answer (safety filter) → need to bypass
  - Model hallucinates non-existent options → need option-list constraint
  - Few-shot examples leak test answers → use disjoint train/test split
```

### Stage 5: Fine-Tuning (Optional, GPU-Intensive)

```
Only if prompt engineering plateaus AND rules allow fine-tuning:

Approach 1 — LoRA/PEFT (parameter-efficient):
  GPU: 8-16GB for 7B model
  Time: 2-4 hours for 1000 examples
  Gain: +2-5% accuracy typical
  Risk: overfit to training distribution

Approach 2 — Full SFT (supervised fine-tuning):
  GPU: 40-80GB for 7B model (multi-GPU or A100)
  Time: 4-8 hours
  Gain: +5-10% accuracy
  Risk: catastrophic forgetting

Approach 3 — RL alignment (DPO/GRPO):
  GPU: same as SFT
  Time: 4-8 hours additional
  Gain: +1-3% over SFT
  Risk: reward hacking

Decision: LoRA if GPU-limited, SFT if competitive, RL only if top 10%.
```

### Stage 6: Submission Packaging

```
For code competitions (no internet):

□ Model weights saved as Kaggle dataset
□ Tokenizer bundled
□ Inference script uses only offline dependencies
□ All imports available in Kaggle environment (pip install --no-index)
□ Notebook runs within time limit (test with small subset first)
□ Output format matches submission specification EXACTLY

Common failures:
  - Internet disabled → pip install fails → bundle wheels as dataset
  - Model too large → use quantization (FP8, INT4)
  - Inference too slow → batch queries, use vLLM
```

## Competition-Specific Strategies

### MCQ/Knowledge Tasks (Industrial Automation T1)
- Base model: Qwen 2.5-7B (fast, good at knowledge)
- Prompt: few-shot with domain examples
- Key: FMEA knowledge base as system prompt (if rules allow)
- Pitfall: metadata leakage (old test set had metadata, new doesn't)

### Agent/Interactive (ARC-AGI-3)
- Base model: Qwen 3.6-27B FP8 (strong reasoning)
- Architecture: Python REPL + segmentation tool + multimodal image
- Key: harness design > model choice (per Tufa Labs Duck writeup)
- GPU: mandatory (27B model, 96GB VRAM, 9h runtime)

### Security/Red-Team (ai-agent-security)
- Base model: GPT-OSS or Gemma (competition-specified)
- Architecture: attack search algorithm + sandboxed agent interaction
- Key: search strategy (fuzzing, evolutionary, Go-Explore)
- GPU: needed for target model inference during development

## Evidence

- Industrial Automation T1: FMEA knowledge base + parser (0.28 LB, metadata removed)
- ARC-AGI-3: Duck harness studied (1.21% LB winner, Qwen 3.6-27B + REPL)
- LLM benchmark: closed-book constraint means model internal knowledge is key
- Prompt engineering typically contributes 80% of score; fine-tuning adds 20%
