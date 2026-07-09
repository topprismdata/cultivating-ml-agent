---
name: code-competition-artifact-pipeline
description: |
  Use when: (1) Entering a Kaggle Code Competition that requires notebooks with
  no internet access, (2) Need to fork a public baseline that depends on external
  artifact datasets (pre-trained models, feature matrices, wheels), (3) Your fork
  fails with "module not found" or "file not found" errors, (4) Need to identify
  and attach the correct Kaggle datasets as notebook inputs. Key lessons: search
  kaggle datasets list for artifact bundles, check kernel-metadata.json
  dataset_sources field, the notebook's find_artifacts() function reveals expected
  paths. Validated on ROGII (7159→Pipeline A), Biohub (LB810 fork), NeuroGolf
  (7228 fork).
---

# Code Competition Artifact Pipeline

## Problem
Kaggle Code Competitions (no internet, notebook-only submission) often have
community baselines that depend on **external artifact datasets**:
- Pre-trained model weights (pickle/onnx)
- Feature matrices (CSV/numpy)
- Custom library wheels (.whl)

Forking these baselines fails unless you identify and attach ALL required
datasets. The error messages are often cryptic ("file not found", "module
not found").

## Solution

### Step 1: Read the Notebook's find_artifacts() Function

Most community baselines have a `find_artifacts()` or similar function that
searches `/kaggle/input/` for specific directory structures:

```python
def find_artifacts():
    candidates = [
        "/kaggle/input/datasets/author/artifact-name",
        "/kaggle/input/artifact-name",
    ]
    # ... searches for repo/, weights/, wheels/, data/train.csv
```

This reveals: **exact dataset names**, **expected file structure**, and
**required Python packages**.

### Step 2: Search for Artifact Datasets

```bash
kaggle datasets list --search "competition-name artifacts"
kaggle datasets list --search "author-name"
```

Look for high-download datasets (500+ downloads = canonical bundle).

### Step 3: Attach ALL Required Datasets

In kernel-metadata.json:
```json
{
  "dataset_sources": [
    "author/competition-artifacts",
    "author/library-wheel"
  ]
}
```

**Common missing datasets** (from real failures):
| Competition | Missing Dataset | Error |
|------------|----------------|-------|
| ROGII | ravaghi/wellbore-geology-prediction-artifacts | "data/train.csv not found" |
| ROGII | phongnguyn23021656/koolbox-offline | "No module named 'koolbox'" |
| Biohub | thibautgoldsborough/cellmot-baseline-artifacts | "repo/ not found" |
| NeuroGolf | (embedded in notebook, no external needed) | — |

### Step 4: Pipeline-Trim for Crashing Components

If the notebook has multiple pipelines (A/B) and one crashes:
- Pipeline A succeeds but Pipeline B errors → kernel ERROR → no submission scored
- Fix: **delete Pipeline B cells**, keep only Pipeline A + final submission write
- Alternative: wrap Pipeline B in try/except

**ROGII case study**: Pipeline A (OOF RMSE 10.38) succeeded but Pipeline B
(lik-PF) crashed. Trimming Pipeline B cells made the kernel COMPLETE, allowing
the submission to be scored.

### Step 5: CPU vs GPU Mode

- CPU kernels: no GPU quota cost, but slower (inference-only mode for some notebooks)
- GPU kernels: faster, but 30h/week quota limit shared across all competitions
- Some notebooks REQUIRE GPU (e.g., Biohub UNet, ARC-AGI Duck harness)

## Anti-Patterns

| Anti-Pattern | Fix |
|-------------|-----|
| Fork without reading find_artifacts() | Read the function to find dataset names |
| Attach only 1 dataset when 2+ needed | Check ALL import/glob patterns |
| Submit ERROR kernel expecting partial score | Code competitions don't score ERRORs |
| Keep crashing Pipeline B cells | Delete them or wrap in try/except |
