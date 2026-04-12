---
name: kaggle-data-format-first
description: |
  Prevent wasted research by verifying Kaggle competition data format BEFORE investing
  in RAG, technical planning, or model architecture design. Use when: (1) Starting any
  new Kaggle competition, (2) Competition name/size is ambiguous about data format,
  (3) Planning to do extensive research before implementation, (4) Download size doesn't
  match expected data structure. Critical for competitions where name suggests one format
  (e.g., "3D Surface Detection") but actual data is different (e.g., 2D images).
---

# Kaggle Data Format Verification Before Research

## Problem

Competition names and file sizes can be misleading. Investing in RAG research, technical
planning, or model architecture design before verifying the actual data format leads to
significant wasted effort when assumptions don't match reality.

**Real example**:
- Competition: "vesuvius-challenge-surface-detection"
- Expected (from RAG): 3D TIFF stacks, TopoScore, 128³ patches
- Actual data: 2D grayscale images (320×320), binary masks
- Waste: Hours of RAG research on wrong problem

## Context / Trigger Conditions

**Use this skill when**:
- Starting ANY new Kaggle competition
- Competition name is ambiguous about data dimensionality (2D vs 3D)
- Data size suggests one format but could be another
- Planning to do RAG research or extensive technical planning
- File extensions are generic (.tif, .png, .npy could be anything)

**Red flags**:
- Competition name mentions "3D", "volume", "surface" but you haven't verified
- Large download size (>5GB) but unsure what format it actually is
- Multiple data directories with unclear purpose (train_images vs train vs train_data)

## Solution

### Phase 1: Quick Format Check (Before ANY Research)

**Step 1**: Download only a sample first
```bash
# If possible, download just one file to verify format
# Or download full data but check structure immediately

kaggle competitions download -c {competition-slug}
unzip {competition-file}.zip
```

**Step 2**: Verify data structure in <5 minutes
```python
import os
from PIL import Image
import numpy as np

# Quick check script
data_dir = "path/to/unzipped/data"

# What files exist?
print("Directories:", [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print("Files:", [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))][:10])

# Check dimensions of first sample
samples = []
for root, dirs, files in os.walk(data_dir):
    for f in files:
        if f.endswith(('.tif', '.png', '.jpg', '.npy')):
            path = os.path.join(root, f)
            if f.endswith('.npy'):
                data = np.load(path)
            else:
                data = np.array(Image.open(path))

            print(f"Sample: {f}")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Range: [{data.min()}, {data.max()}]")

            samples.append({
                'path': path,
                'shape': data.shape,
                'dtype': str(data.dtype)
            })

            if len(samples) >= 3:
                break
    if len(samples) >= 3:
        break

# Determine data type
if all(len(s['shape']) == 2 for s in samples):
    print("✓ Data Type: 2D Images")
elif all(len(s['shape']) == 3 for s in samples):
    print("✓ Data Type: 3D Volumes")
else:
    print("⚠ Mixed or irregular dimensions")
```

**Step 3**: Confirm with competition description
```bash
# Read competition description on Kaggle
# Check Data tab for file descriptions
# Look at sample notebooks from other participants
```

### Phase 2: Match Format to Approach

| Data Format | Typical Approach | Red Flags to Avoid |
|-------------|------------------|-------------------|
| 2D Images | U-Net 2D, ResNet, segmentation | ❌ Using 3D architectures |
| 3D Volumes | nnU-Net 3D, 3D U-Net, V-Net | ❌ Flattening to 2D loses context |
| Tabular | XGBoost, LightGBM, CatBoost | ❌ Using CNN/RNN unnecessarily |
| Time Series | LSTM, GRU, Temporal Fusion | ❌ Treating as i.i.d. samples |
| Text | Transformers, BERT, RoBERTa | ❌ Using bag-of-words |

### Phase 3: Proceed with Confidence

**Only AFTER verifying data format**:
1. Do RAG research with correct context
2. Design appropriate architecture
3. Plan training strategy
4. Set up evaluation metrics

## Verification

**Success criteria**:
- ✅ You know exact dimensions (2D vs 3D vs 4D)
- ✅ You know data type (image, volume, tabular, text)
- ✅ You know number of channels/samples
- ✅ You verified labels match expectations
- ✅ Research/planning now matches actual data

**Failure signs** (means you need to re-verify):
- ❌ Using 3D CNNs on 2D data
- ❌ Researching TopoScore for binary masks
- ❌ Planning patch extraction for already-small images
- ❌ RAG answers don't match your data structure

## Example

**Bad workflow** (what happened):
1. ❌ Saw competition name "vesuvius-challenge-surface-detection"
2. ❌ Assumed 3D based on name
3. ❌ Did extensive RAG research on 3D surface detection
4. ❌ Created technical plan for 3D nnU-Net
5. ❌ Downloaded 24GB data
6. ❌ Discovered data is 2D segmentation
7. ❌ Wasted hours on wrong approach

**Good workflow** (what should have happened):
1. ✅ Saw competition name
2. ✅ Downloaded data FIRST (5 minutes)
3. ✅ Ran quick format check script (2 minutes)
4. ✅ Discovered: 2D grayscale (320×320), binary masks
5. ✅ Adjusted research query: "Vesuvius Challenge 2D segmentation"
6. ✅ Created appropriate 2D U-Net plan
7. ✅ Total time saved: Several hours

## Notes

**Why this happens**:
- Competition names are marketing, not technical specs
- "Surface Detection" could mean 2D edge detection OR 3D surface reconstruction
- File size (24GB) could be: many 2D images OR fewer 3D volumes OR compressed videos
- Multiple Kaggle competitions may have similar names but different tasks

**Time investment**:
- Format check: 5-10 minutes
- Cost of skipping it: 2-10 hours of wasted research

**Integration with RAG**:
- Always verify data format FIRST
- Then use verified format in RAG queries
- Example: "2D grayscale segmentation for papyrus scrolls" NOT "Vesuvius Challenge 3D"

**Related skills**:
- `kaggle-competition-best-practices` - Overall workflow
- `kaggle-top-performer-replication` - After format is verified
- `kaggle-reid-submission-workflow` - Task-specific workflows

## References

- [GitHub - jayinai/how-to-kaggle](https://github.com/jayinai/how-to-kaggle) - Kaggle workflow research
- [Kaggle Competitions Documentation](https://www.kaggle.com/docs/competitions) - Official competition guidelines
- [Towards Data Science - Organizing Code for Kaggle](https://towardsdatascience.com/organizing-code-experiments-and-research-for-kaggle-competitions/) - Code organization best practices
