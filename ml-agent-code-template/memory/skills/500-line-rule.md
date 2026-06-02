# 500-Line Rule for Skills

> Skills must stay under 500 lines. Larger skills bloat context and reduce activation quality.

## Why 500 Lines

- **Context health**: Claude loads SKILL.md into context on activation. 500 lines ≈ 4000 tokens. Skills >500 lines eat 10-15% of context window on activation.
- **Activation quality**: Smaller skills activate more often (less perceived cost).
- **Maintenance**: 500-line files are easier to keep current and review.
- **Evidence**: diet103/claude-code-infrastructure-showcase (9.7k stars) explicitly uses this rule.

## The Pattern

```
skill-name/
├── SKILL.md              # Main, <500 lines, high-level guide
└── resources/            # Optional, only loaded when needed
    ├── topic-1.md        # <500 lines each
    ├── topic-2.md
    └── topic-3.md
```

**Progressive disclosure**: Claude loads SKILL.md first, only loads resources if the task requires them.

## When a Skill Should Be Split

Split when ANY of these:
- File > 500 lines
- File has > 5 distinct topics/sections
- A section is rarely used (but useful in edge cases)
- Different users would only need different parts

## How to Split

```bash
# 1. Create resources/ dir
mkdir -p skill-name/resources

# 2. Move deep content into resource files
# Keep in SKILL.md: front matter, when-to-use, core rule, key examples
# Move to resources/: detailed walkthroughs, edge cases, code reference

# 3. In SKILL.md, link to resources
# "For detailed walkthrough, see resources/topic-1.md"
```

## Enforced By

`skill_size_check.sh` (PostToolUse hook):
- Warns at > 400 lines (soft)
- Blocks at > 500 lines (hard)
- Suggests the split structure

## Anti-Patterns

- ❌ 1000+ line skill that tries to cover everything
- ❌ Splitting too aggressively (one skill → 10 tiny files)
- ❌ "Catch-all" skills with low signal
- ❌ Duplicating content between SKILL.md and resources/

## Reference

- diet103/claude-code-infrastructure-showcase
- Claude Code skill best practices
