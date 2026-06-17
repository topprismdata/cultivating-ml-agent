---
name: okf-visualize-knowledge
description: |
  Use Google's Open Knowledge Format (OKF) to convert any directory of
  markdown files into a force-directed knowledge graph. Validated on
  `docs/ml-agent-memory/`: 12 concepts / 35 edges rendered in 5 seconds.
  Use when: (1) You have a directory of markdown notes with cross-references,
  (2) You want to visualize how your concepts/skills/lessons connect,
  (3) You want to add type-driven coloring (Principle vs Skill vs Lesson
  vs Competition vs Bundle), (4) You want self-contained HTML output
  that runs offline.
---

# OKF Visualize Knowledge

## Problem

Knowledge bases grow organically as folders of markdown files with cross-references. Without visualization:

- Hard to spot which concepts are isolated (zero cross-refs)
- Hard to see which skills have the most evidence
- Hard to communicate "here's what I know" to teammates
- Manual diagrams are stale the moment you write them

**Reality**: A 5-second command turns your notes into an interactive graph.

## Context / Trigger Conditions

Use this skill when:
- You have a directory of `.md` files with **at least one cross-link** (`[text](other.md)`)
- The files have a consistent organization (subdirectories, or flat)
- You want to **audit** which concepts are orphans
- You want a **shareable, offline HTML** view

**Don't use**:
- Single-file notes (no cross-refs → no graph)
- Confidential knowledge (the HTML has all content embedded)

## Solution: 5-Minute Pipeline

### Step 1: Install OKF tools

```bash
git clone https://github.com/GoogleCloudPlatform/knowledge-catalog.git
cd knowledge-catalog/okf
pip install -e .        # installs enrichment-agent CLI
```

### Step 2: Add YAML frontmatter (optional but recommended)

Without frontmatter, OKF treats every file as a "Document" with no type. Add minimal frontmatter:

```markdown
---
type: Skill
title: My Skill Name
description: One-line summary.
tags: [skill, kaggle]
timestamp: 2026-06-17T00:00:00Z
---

# My Skill

Body text with [cross-links](other-file.md)...
```

### Step 3: Add `index.md` to each subdirectory (optional)

Each subdirectory gets a hub page:

```markdown
---
type: Index
title: Subdir Name
description: What lives here.
---

# Subdir Name

* [file-one](file-one.md)
* [file-two](file-two.md)
```

### Step 4: Run visualize

```bash
enrichment-agent visualize \
  --bundle /path/to/your/notes \
  --out /tmp/my-graph.html
```

Output: a self-contained HTML file (~50 KB) you can `open` or share.

### Step 5: Iterate

Add a new concept → re-run visualize → see how the graph grew. Orphan concepts (no cross-refs) become obvious in the graph — fix them by adding links.

## What OKF Generates

```
Wrote 12 concept(s), 35 edge(s), 64117 bytes → /tmp/my-graph.html
```

- **concept(s)**: markdown files with at least one frontmatter or body content
- **edge(s)**: markdown links between files (counted as edges)
- **bytes**: HTML output size (~5 KB per concept)

## Type-Driven Coloring

The viewer colors nodes by `type:` frontmatter field:

| Type | Color |
|---|---|
| `Bundle` / `Index` / `Dashboard` | structural nodes (gray/blue) |
| `Principle` / `Principle Set` | wisdom (mauve) |
| `Skill` / `Lesson` | actionable (coral) |
| `Competition` / `Experiment` / `Model` / `Submission` | ML domain (green/cyan) |
| Unknown | default |

Use **descriptive, self-explanatory type values** — consumers (visualize, future readers) handle unknown types gracefully.

## Validation

`docs/ml-agent-memory/` is the working example:
- 12 concepts / 35 edges / 64 KB HTML
- 4 subdirectories (`competitions/`, `experiments/`, `skills/`, `lessons/`)
- 8 different type values used
- Renders in < 5 seconds on M-series Mac

```bash
cd ~/projects/cultivating-ml-agent/docs/ml-agent-memory
enrichment-agent visualize --bundle . --out /tmp/viz.html
open /tmp/viz.html
```

## Why OKF vs Alternatives

| Tool | Pro | Con |
|---|---|---|
| OKF + visualize | Plain markdown, git-versioned, offline | Manual frontmatter |
| Obsidian | Polished UI, daily use | Tied to Obsidian app, no offline HTML |
| MkDocs | Mature, themes | Static site, no interactive graph |
| Logseq | Block-based, Roam-like | Different mental model |

OKF wins when you want a **vendor-neutral format** with **git-versioned history** and a **single static-HTML viewer**.

## Real-World Usage

Validated during S6E2 AutoGluon rerun (2026-06-14):
- Discovered ml-agent-memory had 0 OKF edges before migration
- Added frontmatter + cross-links → jumped to 6/24 in 10 minutes
- Subsequent S6E2 lesson extractions → 12/35 in another 30 minutes

Each new concept added at < 5 minutes total (write frontmatter + cross-link to existing concepts).

## Notes

**Type values are free** but pick descriptive ones. The OKF repo's
[issue #60](https://github.com/GoogleCloudPlatform/knowledge-catalog/issues/60)
proposes ML-canonical types (`Experiment`, `Model`, `Submission`, `Lesson`,
`Competition`, `Bundle`). Use those for ML knowledge; pick your own for other domains.

**File-relative links only**. The OKF viewer rewrites absolute paths to GitHub-friendly
relative paths (PR #45 in OKF repo).

## Related Skills

- `kaggle-data-format-first` — verify data format BEFORE research (similar "verify first" philosophy)
- `autogluon-first` — get a strong baseline in 15 min before iterating
- `ml-sweet-spot` — when to stop optimizing

## References

- [GoogleCloudPlatform/knowledge-catalog](https://github.com/GoogleCloudPlatform/knowledge-catalog) — OKF source
- [OKF v0.1 SPEC](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
- PR #6 in `topprismdata/cultivating-ml-agent` — OKF migration of `docs/ml-agent-memory/`