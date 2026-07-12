# Contributing to Cultivating ML Agent

We welcome contributions! Here's how to help:

## Adding a New Skill

1. Read existing skills in `skills/examples/` for format reference
2. Create a new directory: `skills/examples/your-skill-name/SKILL.md`
3. Follow the skill template (`templates/knowledge-skill.md`)
4. Include:
   - **Trigger conditions**: When to activate this skill
   - **The rule**: One clear sentence
   - **How to apply**: Step-by-step
   - **Code example**: Working code snippet
   - **Anti-patterns**: What NOT to do
5. Submit a PR

## Adding a Dead End

1. Add to `memory/feedback_no_recheck_confirmed_dead.md`
2. Include: direction, failure mode, evidence (experiment + numeric result)

## Skill Quality Standards

- **Validated**: Skill must be tested on at least 1 real competition
- **Specific**: Trigger conditions must be unambiguous
- **Actionable**: Clear code example that can be directly applied
- **Transferable**: Principle should apply beyond one specific dataset

## Commit Message Format

```
feat: add <skill-name> — <one-line description>
fix: correct <what was wrong> in <skill-name>
docs: update <what> in <file>
```
