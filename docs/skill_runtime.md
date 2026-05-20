# Skill Runtime

Each skill is a folder with:

- `manifest.yaml`
- `SKILL.md`
- optional `scripts/`
- optional `config_templates/`

The MVP loads:

- L0: manifest metadata.
- L1: `SKILL.md`.

`SkillExecutor` only runs allowlisted commands. Current allowlist:

- `deeplens-adapter/run_mock_psf`

The executor does not run arbitrary shell commands. `SkillValidator` checks expected artifacts, metrics, and claim-evidence shape.

## SkillMemory

After a run, `SkillMemoryManager` compiles skill traces into:

- used run IDs
- success and failure counts
- success rate
- emitted artifact types
- commands
- common failures
- best practices

`SkillRouter` keeps keyword routing as the first pass, then uses SkillMemory success rate and preferred cues to sort matching skills.
