# AGENTS.md

## Project Goal

This repository implements OptiResearch Agent, a skill-augmented and memory-grounded autonomous research agent for differentiable optical design.

## Engineering Rules

- Prefer small, testable changes.
- Preserve existing project structure unless clearly broken.
- Do not require real DeepLens for MVP; use MockDeepLensAdapter.
- All agent actions that produce outputs must write Meta-Trace entries.
- All promoted results must register artifacts.
- All claims must be linked to evidence or marked unsupported.
- Do not execute arbitrary shell commands from skills; use allowlisted commands only.
- Do not hardcode machine-specific absolute paths.
- Use deterministic seeds for mock simulations.
- Add or update tests for every new module.

## Test Commands

- `python -m pytest`
- `python -m optiresearch.cli init-db`
- `python -m optiresearch.cli run-mvp --objective "Design a mock EDOF-HSI optical encoder"`
