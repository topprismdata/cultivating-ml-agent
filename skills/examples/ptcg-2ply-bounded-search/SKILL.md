---
name: ptcg-2ply-bounded-search
description: |
  Augment a BC Transformer agent with 2-ply bounded search using the cg library's
  SearchBegin/SearchStep API. Validated 2026-07-30 on Pokemon TCG AI Battle:
  search-enhanced agent submitted as v8 (LB pending). Use when: (1) you have a
  trained BC policy/value model, (2) the cg library exposes a search API
  (SearchBegin/SearchStep/SearchRelease), (3) you want lookahead beyond pure
  reaction-based action selection. Key risks: hidden-info sampling for unknown
  opponent state (use SNORLAX=1072 pad by convention), time budget per
  decision (N=5 samples × ~30 search_steps ≈ 1-3s), value-head quality
  unknown without calibration. Start with small N_SAMPLES and short rollout,
  tune up.
---

# Pokémon TCG — 2-ply Bounded Search on BC Agent

## Why search on top of BC?

Pure BC is reaction-based: it picks the highest-probability action at the
current state. It cannot look ahead. The 1st-place LB 1204 model uses
"RL + bounded search" — search is the structural advantage.

A BC model already has a **value head** (trained alongside the policy head).
Re-encoding a hypothetical future state and reading the value gives a leaf
evaluation — usable for any-tree, expectimax, or simple best-of-K scoring.

## Algorithm (validated pipeline, v8)

```python
def agent(obs_dict, deck, model):
    obs = to_observation_class(obs_dict)
    if obs.select is None: return deck
    options = obs.select.option
    if not options: return []

    # 1. Root forward (one shared pass)
    enc_idx, enc_val, enc_off, opt_t = encode(obs, deck, options)
    with torch.no_grad():
        root_logits, _ = model(enc_idx, enc_val, enc_off, opt_t)
    root_probs = sigmoid(root_logits)  # [N_opt]

    # 2. For each candidate option i:
    scores = []
    for i in range(N_opt):
        sample_scores = []
        for _ in range(N_SAMPLES):              # = 5
            opp = sample_opponent_hidden(obs, deck)  # SNORLAX pad
            ss = search_begin(obs, ..., opp)         # root branch
            # 2a: my move i
            s1 = search_step(ss.search_id, [i])
            # 2b: opponent's BC-greedy response
            opp_choice = bc_greedy(model, s1.observation, deck)
            # 2c: opponent's move
            s2 = search_step(s1.search_id, opp_choice)
            # 2d: leaf value
            leaf_v = encode_and_value(model, s2.observation, deck)
            sample_scores.append(root_probs[i] + GAMMA * leaf_v)
            search_release(ss.search_id)
        scores[i] = mean(sample_scores)

    # 3. top-k by score
    return topk_indices(scores, n_select)
```

## Parameters (validated defaults)

| Param | Value | Notes |
|---|---|---|
| `N_SAMPLES` | 5 | hidden state samples per option |
| `GAMMA` | 0.5 | leaf-value discount (lower = trust BC more) |
| Opponent model | BC-greedy | same model, sigmoid top-1 |
| Hidden state padding | SNORLAX=1072 unknown Pokemon, id=1 basic energy | convention from `probe_search.py` |
| `n_select` | `max(obs.select.maxCount, minCount)` | same as v6/v7 |

## API surfaces (must use EXACTLY)

```python
# ctypes — battle_ptr is from AgentStart(), NOT from battle_start()
ss = search_begin(
    obs,                                    # Observation dataclass
    your_deck       = list[int],
    your_prize      = list[int],
    opponent_deck   = list[int],
    opponent_prize  = list[int],
    opponent_hand   = list[int],
    opponent_active = list[int],             # face-down unknowns are SNORLAX
    manual_coin     = False,
)
# ss.searchId is an int

next_state = search_step(ss.searchId, [option_indices])  # returns SearchState
# SearchState.searchId may change between steps — use whatever it returns

search_release(ss.searchId)  # free a branch
```

Error codes: `1` invalid id, `2` already released, `3` battle ended,
`4` select length bad, `5` option out of range, `6` duplicate options,
`30` agent_ptr broken (need restart).

## Time budget (Kaggle)

- Per decision: `N_SAMPLES × N_options × (1 step_me + 1 step_opp + 1 forward) × ~50ms`
  = `5 × 10 × 3 × 50ms` ≈ 7.5s worst case (10 options)
- Real case typically ~1-3s per decision
- Search API is in-process (same battle_ptr), no IPC overhead
- Model forward is CPU-only on Kaggle (`_DEVICE = "cpu"`) — this dominates

## Pitfalls (verified)

1. **`agent_ptr` is module-level global**, initialized lazily on first
   `search_begin`. Persists across battles; do NOT recreate.
2. **`SearchState.searchId` may change between `search_step` calls** — always
   use the latest returned id.
3. **`Observation` dataclass** must be passed (not raw dict) to `search_begin`.
   Convert with `to_observation_class(obs_dict)`.
4. **Padding cards**: SNORLAX (1072) for unknown Pokemon, id=1 (basic energy)
   for unknown non-Pokemon cards. From `my_agent/probe_search.py` convention.
5. **Hidden-info sampling is noisy** — value head may be unreliable. With
   GAMMA=0.5 the BC root prob dominates the score, which is safer.
6. **Catch all exceptions** in the per-option search loop — fall back to
   `sigmoid(root_logit)` so a search failure doesn't degrade the agent.

## Anti-patterns

- Do NOT use `argmax(probs)` without `topk` semantics — the model is
  multi-label (sigmoid + threshold), not categorical (softmax + argmax).
- Do NOT skip `search_release` — the underlying memory pool is finite.
- Do NOT extend search to 3-ply without time-budget testing on Kaggle.

## References

- `C:/Users/ghb/ZCodeProject/ptcg_agent/my_agent/probe_search.py` —
  the only working example of `search_begin` / `search_step` / `search_end`
- `C:/Users/ghb/ZCodeProject/ptcg_agent/cg-lib-download/cg/api.py` —
  Python wrappers + dataclasses (`SearchState`, `Observation`)
- `C:/Users/ghb/ZCodeProject/cptcg_agent/submission_search/main.py` —
  full submission implementation (v8, this skill's reference)
- Related skill: `ptcg-bc-large-model-submission` (BC training pipeline
  that produces the policy/value model used here)
