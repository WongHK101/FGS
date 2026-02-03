# Codex Agent Rules (FGS)

## Hard constraints
- Backward-compatible is mandatory: new flags default OFF; when disabled, behavior and outputs should match baseline.
- One-file-per-delivery unless user explicitly requests multiple files.
- No new dependencies unless user explicitly approves.
- Avoid formatting-only diffs (no reorder imports / refactor unrelated code).
- After each change: provide
  1) change summary (3-6 bullets)
  2) compatibility checklist
  3) minimal self-check commands (PowerShell or python -c)

## Repo context
- OS: Windows + PowerShell
- Python: conda env (fgs)
- Typical checks: `python -c "import py_compile; ..."`
