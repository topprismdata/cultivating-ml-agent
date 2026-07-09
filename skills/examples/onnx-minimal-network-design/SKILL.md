---
name: onnx-minimal-network-design
description: |
  Use when: (1) Competing in ONNX-based competitions (NeuroGolf, model compression),
  (2) Need to design minimal neural networks as ONNX graphs, (3) Want to replace
  verbose agent-generated circuits with compact hand-designed nets. Key principles:
  single-node graphs have zero memory cost (output is free), Conv is a linear
  classifier (cannot separate nonlinear rules), Gather implements color
  permutations (10 params), use LogisticRegression to solve Conv weights for
  local 3×3 rules. Sparse initializers are rejected by ONNX strict checker for
  Conv nodes. Validated on NeuroGolf 2026 (7228 baseline, independent toolkit).
---

# ONNX Minimal Network Design

## Problem
Neural network golf competitions require the **smallest possible ONNX network**
that correctly implements a transformation. Score = `max(1, 25 - ln(memory + params))`.

Smaller networks → exponentially higher scores. The key insight: **memory
dominates cost** (intermediate tensor bytes), not parameters.

## Solution

### Core Scoring Formula
```
memory = Σ(bytes of all intermediate tensors)  # input/output are FREE
params = Σ(initializer element counts)
score  = max(1, 25 - ln(memory + params))
```

### Design Principle 1: Single-Node = Zero Memory

A single-node graph (1 Conv/Gather/Transpose) has **zero intermediate memory**
(output tensor is free). Only params count.

**Implication**: A 1-node Conv with 900 params (cost=900, score=18.2) beats a
4-node net with 200 params + 5000 memory (cost=5200, score=16.3).

### Design Principle 2: Conv is a Linear Classifier

A Conv node maps 3×3 one-hot neighborhoods → output channels. This is a
**linear classifier** on 90-dim features (9 positions × 10 colors).

To find Conv weights for a local rule:
1. Collect all (3×3 neighborhood, output color) pairs from training data
2. Verify the rule is deterministic (same neighborhood → same output)
3. Train LogisticRegression (multi-class, C=1e4, solver='newton-cg') on the pairs
4. Extract coef_ as Conv W[10,10,3,3], intercept_ as B[10]
5. Verify ONNX runtime output matches ground truth on ALL examples

**Limitation**: If LogReg accuracy < 100%, the rule is NOT linearly separable.
Need nonlinear ops (Cast/Equal/Where) — a single Conv cannot implement it.

### Design Principle 3: Gather for Color Permutations

A pixel-wise color map (e.g., swap color 3↔4) is a 1-node Gather:
- idx = [0,5,6,4,3,1,2,7,9,8] (10-element lookup table)
- Gather(input, idx, axis=1) maps each one-hot vector to the permuted one
- Cost: 10 params, 0 memory → score ≈ 22.7

### Design Principle 4: Sparse Initializers REJECTED

Converting dense sparse weights to `sparse_initializer` reduces params but
**onnx.checker.check_model(full_check=True) rejects sparse Conv weights**:
```
W typestr: T, has unsupported type: sparse_tensor(float)
```

Only use sparse for ops that explicitly support it (not Conv/MatMul in practice).

## Independent Toolkit Architecture

```python
# Grid encoder: ARC grid [[c,...]] → one-hot [1,10,30,30] float32
grid_to_onehot(grid, size=30)

# ONNX builders:
design_gather_colormap(color_map)  # 1-node Gather, 10 params
design_transpose()                  # 1-node Transpose, 0 params  
design_border(color=8)             # 1-node Conv border detection

# Verifier: onnxruntime on ALL train + arc-gen examples
verify(model_bytes, task_json)

# Scorer: compute params + memory → points
score_model(model_bytes)
```

## Evidence (NeuroGolf 2026)
- task171 border: independently designed 1-Conv (19/19 correct, matches existing bundle)
- task222: LogReg solver achieved 99.95% (not 100% — rule is nonlinear)
- 87 tasks had sparse-able weights but sparse rejected by strict checker
- 400-task scan: all simple rules already 1-node in community bundle
