# Karpathy Guidelines

This repo uses a compact set of operating rules derived from Andrej Karpathy's comments on common LLM coding mistakes.

Use these rules as default behavior for non-trivial work.

## 1. Think Before Coding

- State assumptions explicitly.
- Do not silently pick an interpretation when the request is ambiguous.
- Surface tradeoffs and missing information early.
- Ask for clarification when the task is genuinely unclear.

## 2. Simplicity First

- Prefer the smallest solution that solves the problem.
- Do not add speculative abstractions, knobs, or fallback paths.
- Do not over-engineer for edge cases that are not credible for the task.
- If the implementation feels large for the problem, simplify it.

## 3. Surgical Changes

- Change only what the task requires.
- Do not refactor adjacent code just because you touched the file.
- Keep the existing style unless there is a direct reason to change it.
- Remove only the code made unused by your own edits.

## 4. Goal-Driven Execution

- Define success criteria before implementing.
- Prefer tests or concrete verification steps over vague completion claims.
- For multi-step tasks, work in small verifiable increments.
- Keep iterating until the target behavior is actually proven.

## Practical Use

For this repository:

- Use these guidelines when editing code, reports, prompts, or docs.
- They are a bias toward clarity, minimalism, and verifiable outcomes.
- They do not override the repo's domain-specific rules in `docs/market_regime_detect_specs.md`.
