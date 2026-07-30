---
name: ptcg-bc-large-model-submission
description: |
  Submit a BC (behavioral cloning) Transformer agent for the Pokémon TCG AI
  Battle competition. Validated 2026-07-30: best.pth (15-epoch BC Large, d_model=256)
  → **public LB 600.0** (vs prior 165.6 — 3.6× improvement from using the best
  checkpoint instead of a stale submission). Use when: (1) training a BC agent
  on Kaggle Pokémon TCG episodes, (2) submitting a tar.gz agent for the
  competition, (3) scaling BC data to 80K+ games. Critical pitfall: your highest
  validation WR checkpoint is almost certainly NOT what you last saved to your
  submission directory — always verify the submission's model_bc.pth matches the
  best checkpoint before submitting. Stream-sampling (load 50K per part then
  drop) prevents OOM when scaling to 500K+ samples on a 64GB machine.
---

# Pokémon TCG — BC Large Model Submission Pipeline

## The Core Lesson (validated)

The single biggest jump (165 → 600) came from **using the right checkpoint**.
Training produces `checkpoints_bc_large/best.pth` (66 MB, 16.6M params).
Submissions require `model_bc.pth` (v6 path). Forgetting to copy `best.pth`
→ `submission/model_bc.pth` is the #1 reason scores stay low after training.

| Submission | Source | LB |
|---|---|---|
| v6 | stale small-model checkpoint | 165.6 |
| **v7** | **`checkpoints_bc_large/best.pth`** | **600.0** |

## Pipeline (validated end-to-end)

### Step 1: Scale BC data via Kaggle dataset snapshots

Episodes are published daily as `kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD`.
Each snapshot is ~5000 games (not 200 — that was the pitfall of per-file
downloading). Use whole-snapshot download (one zip per day), not per-file:

```bash
for date in 2026-06-20 2026-06-22 ...; do
  kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-$date -p bc_zips
  unzip bc_zips/...zip -d bc_data/$date
done
# Total: 80K+ games → ~6.4M (state, action) pairs
```

### Step 2: Stream extract to part files (avoids OOM)

`extract_bc_data.py` must NOT accumulate the full sample list in memory before
saving. With 6.4M samples at ~10KB each = 64 GB → instant OOM. Write part
files every 50K samples, then `del part_data` after each:

```python
PART_SIZE = 50_000
for i, rf in enumerate(replay_files):
    samples = extract_from_replay(rf)
    batch.extend(samples)
    if len(batch) >= PART_SIZE:
        torch.save(batch, f"bc_dataset_part_{part_idx}.pt")
        part_idx += 1; batch = []
torch.save({"n_parts": part_idx, "total": total_samples}, "bc_dataset_meta.pt")
```

### Step 3: Stream-sampling training (avoids OOM at load)

`train_bc_large.py` loading 129 part files → `extend()` into one list = OOM.
Sample per-part to quota (50K / 129 ≈ 400 per part), `del part_data` after each:

```python
per_part_quota = MAX_SAMPLES // len(part_files)
for pf in part_files:
    part_data = torch.load(pf, weights_only=False)
    if len(part_data) > per_part_quota:
        idx = np.random.choice(len(part_data), per_part_quota, replace=False)
        samples.extend([part_data[j] for j in idx])
    else:
        samples.extend(part_data)
    del part_data  # critical: release memory
```

### Step 4: Skip precompute_encoding, encode on-the-fly

Pre-encoding 500K samples × sparse features = OOM in torch tensor staging.
Forward_batch must accept raw samples and call `encode_state_sparse` inline.
Modify `forward_batch` signature to `(model, batch, device, deck)`.

### Step 5: Always copy best.pth → submission/model_bc.pth before packaging

```bash
cp checkpoints_bc_large/best.pth submission_bc_v7/model_bc.pth
cd submission_bc_v7
tar -czf ../submission_bc_v7.tar.gz .
kaggle competitions submit -c pokemon-tcg-ai-battle \
    -f submission_bc_v7.tar.gz \
    -m "v7: best.pth (vs rule 94.7%)"
```

## Training Recipe (matches `checkpoints_bc_large/log.json`)

```
Model:    TransformerAgent(d_model=256, n_heads=4, d_ff=512, n_enc_layers=4)
           → 16.6M params, 66 MB .pth
Loss:     BCE-with-logits, multi-label sigmoid head
Data:     50K-500K samples (from stream-sampled part files)
Train:    AdamW lr=5e-4, cosine warmup 2 epochs, 15 epochs total
Eval:     vs rule agent every epoch, keep best-WR checkpoint
Runtime:  ~3.5 hours on RTX-class GPU
Result:   94.7% WR vs rule, LB 600.0 (vs prior 165.6)
```

## Anti-Patterns (verified pitfalls)

1. **Submitting the wrong checkpoint** — biggest single failure mode. Always
   `cp checkpoints_bc_large/best.pth submission_*/model_bc.pth` before tar.
2. **`extend()` accumulating all part files** — OOM at 6.4M samples. Sample per-part.
3. **`precompute_encoding` of full sample set** — OOM. Use forward_batch on-the-fly.
4. **Per-file download (`kaggle datasets download ... -f file.json`)** — only
   gets 200 of ~5000 files/day. Use whole-snapshot zip download.
5. **Old `bc_encoded_cache_large.pt`** — if you switch training recipes,
   `rm bc_encoded_cache_large.pt` or the cache will mask the new recipe's bugs.

## Architecture Notes

- **cg library (`cg/sim.py`)** exposes `SearchBegin/SearchStep/SearchEnd`
  for forward simulation — this is the API first-place Luca uses for
  "bounded search". The top solutions add search on top of BC/RL policies;
  pure BC/PPO tops out around LB 700-800.
- **`TransformerAgent.forward`** takes sparse encoding indices/values/offsets
  plus dense option features. The encoder is `EmbeddingBag` (sparse) → `TransformerEncoder`
  → cross-attention with option embeddings → per-option sigmoid logit.

## References

- Competition: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- Episode snapshots: `kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD` (daily)
- Top solution reference: Luca (LB 1204, RL + bounded search)
- Local training SOP: `ptcg_agent/TRAINING_SOP_v2.md` (Tony Li 21K-recipe analysis)
- Validation history: this repo's own `checkpoints_bc_large/log.json`
