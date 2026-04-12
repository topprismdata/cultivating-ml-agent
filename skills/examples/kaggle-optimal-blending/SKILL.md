---
name: kaggle-optimal-blending
description: |
  Optimal blending strategy for Kaggle competitions: 80/20 rule (re-ranking vs raw).
  Use when: (1) Re-ranking gives unexpected results, (2) Simple average blending
  underperforms, (3) Single model dominates ensemble. Avoids the trap of
  100% re-ranking or equal-weight ensembles.
---

# Kaggle Optimal Blending: The 80/20 Rule

## Problem
Re-ranking (k-reciprocal, Jaccard, etc.) is powerful but using 100% re-ranked
results can lose valuable signal diversity. Equal-weight blending often underperforms.

## Context / Trigger Conditions
Use when:
- Implementing post-processing for Re-ID, retrieval, or recommendation tasks
- Re-ranking alone doesn't give expected boost
- Ensemble of multiple models shows little improvement
- Kaggle score plateaus despite strong individual models

## Solution

### The 80/20 Optimal Blending Rule

**Key insight from 0.938 mAP solution (Jaguar Re-ID):**

```python
blend_raw_ratio = 0.20  # 20% raw, 80% re-ranked

final_similarity = (
    (1 - blend_raw_ratio) * rerank_score +
    blend_raw_ratio * raw_score
)
```

**Why 80/20?**
- 80% re-ranking: Captures neighborhood structure, removes false positives
- 20% raw: Preserves genuine long-range similarities, maintains diversity
- This ratio emerged as optimal through experimentation (0.938 mAP)

### Implementation

```python
import numpy as np

def optimal_blend(similarity_matrix, reranked_matrix, raw_ratio=0.20):
    """
    Blend re-ranked and raw similarity matrices

    Args:
        similarity_matrix: Original NxM similarity matrix
        reranked_matrix: Re-ranked version (k-reciprocal, etc.)
        raw_ratio: Weight for raw score (default 0.20)

    Returns:
        Blended similarity matrix
    """
    return (1 - raw_ratio) * reranked_matrix + raw_ratio * similarity_matrix
```

### Re-Ranking Parameters (Optimized for 0.938 mAP)

```python
# k-Reciprocal Re-ranking parameters
k1 = 20   # First k for reciprocal neighbors
k2 = 6    # Second k for final re-ranking
lambda_param = 0.2  # Jaccard weight
```

### Complete Post-Processing Pipeline

```python
def post_process_for_reid(query_features, gallery_features):
    """
    Complete post-processing: TTA + QE + Re-ranking + Optimal Blend
    """
    # 1. TTA (Test Time Augmentation)
    # query_features = (f_original + f_flipped) / 2

    # 2. Query Expansion (optional)
    # Expand with top-k gallery samples
    qe_top_k = 3

    # 3. Compute raw similarities
    raw_sim = cosine_similarity(query_features, gallery_features)

    # 4. Re-ranking (k-reciprocal)
    reranked_sim = k_reciprocal_reranking(
        raw_sim,
        k1=k1,
        k2=k2,
        lambda_param=lambda_param
    )

    # 5. Optimal blending (THE KEY!)
    final_sim = optimal_blend(
        raw_sim,
        reranked_sim,
        raw_ratio=0.20  # Don't use 0.0 or 0.5!
    )

    return final_sim
```

## Verification

**Test the blend ratio:**

```python
# Test different ratios
for ratio in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]:
    blended = optimal_blend(raw_sim, reranked_sim, raw_ratio=ratio)
    score = evaluate(blended, val_labels)
    print(f"Ratio {ratio}: {score:.4f}")
```

**Expected pattern:**
- 0.0 (100% re-rank): Good but may over-prune
- 0.10: Better
- **0.20: Optimal** ← Use this!
- 0.25: Slight decline
- 0.50: Significant decline (too much raw)

## Example

**Jaguar Re-ID Competition:**

```python
# Without optimal blending
# Re-ranking only: 0.92 mAP
# Raw only: 0.85 mAP

# With optimal blending (80/20)
final_score = 0.8 * reranked + 0.2 * raw
# Result: 0.938 mAP (+1.8% over re-ranking alone)
```

**H-Blend Strategy (0.944 mAP solution):**
```python
# Single strong model should dominate
weights = {
    'model_0.944': 0.95,  # Dominant!
    'model_0.938': 0.02,
    'model_0.937': 0.02,
    'model_0.930': 0.01,
}
```

This validates the "ensemble-model-correlation-trap" skill.

## Notes

### When to Use 80/20
✅ Person/Animal Re-ID tasks
✅ Image retrieval with re-ranking
✅ Recommendation systems with neighborhood methods
✅ Any metric learning task with post-processing

### When to Adjust the Ratio
- Very clean data: Try 0.15 (more re-ranking)
- Noisy data: Try 0.25-0.30 (more raw signal)
- High threshold requirements: Try 0.10

### Common Mistakes
❌ Using 100% re-ranking (loses diversity)
❌ Using 50/50 blend (loses re-ranking benefit)
❌ Equal-weight ensembles (use 95/5 for strong/weak)
❌ Tuning ratio on test set (use validation only!)

### Related Skills
- `ensemble-model-correlation-trap`: Why strong models should dominate
- `k-reciprocal-reranking`: Implementation details
- `eva02-reid-backbone`: Produces strong features for blending

## References

- Kaggle Jaguar Re-ID: sanidhyavijay24 (0.938 mAP)
- Kaggle Jaguar Re-ID: nina2025 H-Blend (0.944 mAP)
- k-Reciprocal Re-ranking: "Re-ranking Person Re-identification" (CVPR 2017)
