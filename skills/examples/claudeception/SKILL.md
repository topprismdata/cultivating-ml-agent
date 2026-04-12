---
name: claudeception
description: |
  Extract reusable knowledge from work sessions into new Claude Code skills. Use when:
  (1) /claudeception command to review session, (2) "save this as a skill" or "extract a skill",
  (3) "what did we learn?", (4) After non-obvious debugging, workarounds, trial-and-error
  discovery, or counterintuitive solutions. Do NOT trigger for routine fixes or straightforward
  tasks with obvious solutions.
---

# Claudeception

Continuous learning system that extracts reusable knowledge from work sessions into
Claude Code skills. Each extracted skill makes future sessions smarter.

## When to Extract

Extract when you encounter:

1. **Non-obvious solutions** — Required >10 min investigation, not in docs
2. **Error resolution** — Misleading error messages, non-obvious root causes
3. **Workaround discovery** — Tool/framework limitations requiring experimentation
4. **Configuration insights** — Project-specific setups differing from standard
5. **Trial-and-error success** — Multiple approaches before finding what worked

## Dual-Track Classification

Before creating, classify the knowledge:

| Track | When | Template sections |
|-------|------|-------------------|
| **Bug fix** | Defect, failure, error resolution | Problem, Symptoms, Root Cause, Solution, Prevention |
| **Knowledge** | Best practice, pattern, workflow optimization | Context, Guidance, Why This Matters, When to Apply |

This determines the skill's section structure.

## Extraction Process

### Step 1: Overlap Detection (BEFORE creating)

Search existing skills for overlap across 5 dimensions:

| Dimension | What to compare |
|-----------|----------------|
| Problem statement | Same underlying issue? |
| Root cause | Same technical cause? |
| Solution approach | Same fix? |
| Referenced files | Same code paths? |
| Prevention rules | Same advice? |

**Scoring**: Count matching dimensions.

| Overlap | Action |
|---------|--------|
| **High** (4-5 match) | Update existing skill with fresher context |
| **Moderate** (2-3 match) | Create new, add `See also:` cross-reference |
| **Low** (0-1 match) | Create new normally |

**Why**: Two skills describing the same problem will drift apart. Update rather than duplicate.

### Step 2: Research (When Appropriate)

Search the web when the topic involves specific technologies, frameworks, or APIs.
Skip for project-specific internal patterns. Cite sources in a References section.

### Step 3: Structure the Skill

Use this template — adapt sections based on track:

```markdown
---
name: [descriptive-kebab-case-name]
description: |
  [Trigger conditions ONLY. Start with "Use when...". Include specific symptoms,
  error messages, contexts. NEVER summarize the skill's workflow here.]
---

# [Skill Name]

## Problem / Context
[Bug track: What broke. Knowledge track: What situation prompted this.]

## Symptoms / When to Apply
[Bug track: Observable symptoms. Knowledge track: Conditions where this applies.]

## Solution / Guidance
[Step-by-step fix or recommended practice with code examples.]

## Prevention / Why This Matters
[Bug track: How to avoid recurrence. Knowledge track: Impact of following/not following.]

## Notes
[Caveats, edge cases, See also: links to related skills]

## References
[Optional: URLs to official docs or resources]
```

### Step 4: Write Effective Descriptions (CSO)

**Description = Triggering conditions ONLY. Never summarize the workflow.**

Why: Testing revealed that descriptions summarizing workflow cause agents to follow
the description instead of reading the full skill. A description saying "code review
between tasks" caused agents to do ONE review, when the skill required two.

```yaml
# BAD: Summarizes workflow — agent shortcuts to this
description: Use when executing plans — dispatches subagent per task with code review

# GOOD: Trigger conditions only — agent reads the full skill
description: Use when executing implementation plans with independent tasks
```

**Format rules:**
- Start with "Use when..."
- Include specific symptoms, error messages, contexts
- Keep under 500 characters
- Third person (injected into system prompt)
- Include technology names if skill is technology-specific

### Step 5: Discoverability Check

After creating a skill, verify:
1. The description contains keywords someone would search for
2. The name is descriptive (verb-first preferred: `fixing-X`, not `X-fix`)
3. If the project has a CLAUDE.md, consider whether it should reference this skill category

### Step 6: Save

- **Project-specific**: `.claude/skills/[skill-name]/SKILL.md`
- **User-wide**: `~/.claude/skills/[skill-name]/SKILL.md`
- Heavy reference (>100 lines) goes in `references/` subdirectory
- Reusable scripts go in `scripts/` subdirectory

## Quality Gates

Before finalizing:
- [ ] Description is trigger-only, no workflow summary (CSO)
- [ ] Solution verified to work (not theoretical)
- [ ] Specific enough to be actionable
- [ ] General enough to be reusable
- [ ] No sensitive information
- [ ] Overlap check completed (Step 1)
- [ ] No duplicate of existing skill or official docs

## Anti-Patterns

| Anti-pattern | Why it's bad |
|-------------|-------------|
| **Over-extraction** | Mundane solutions don't need skills |
| **Vague description** | "Helps with React" won't surface when needed |
| **Workflow in description** | Agent shortcuts, skips reading full skill |
| **Unverified solution** | Only extract what actually worked |
| **Duplication** | Two skills on same topic drift apart over time |
| **Narrative storytelling** | "In session 2025-10-03 we found..." — too specific |
| **Multi-language examples** | One excellent example beats 5 mediocre ones |

## Retrospective Mode

When `/claudeception` is invoked at session end:
1. Review conversation for extractable knowledge
2. List candidates with brief justifications
3. Prioritize highest-value, most reusable (1-3 per session)
4. Extract using the process above
5. Summarize what was created and why

## Self-Check Prompts

After completing any significant task:
- "Did I just spend meaningful time investigating something?"
- "Would future-me benefit from having this documented?"
- "Was the solution non-obvious from documentation alone?"

If yes to any, extract now.

## Integration with Other Skills

| Skill | Relationship |
|-------|-------------|
| three-layer-wisdom-extraction | Operates ABOVE claudeception — uses claudeception output as Layer 2 input |
| skill-refresh | Maintains claudeception-created skills over time |
| skill-creator | For formal eval/benchmark testing of skills |
