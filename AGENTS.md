# Codex Agent Rules (FGS)

## Hard constraints
- Backward-compatible is mandatory: new flags default OFF; when disabled, behavior and outputs should match baseline.
- For modifications involving a small amount of code, multiple files may be modified at one time. For modifications involving a large amount of code, you shall first report the modification plan to the user (including but not limited to the involved files, modification schemes, etc.), and proceed with the modifications only after the user confirms that the plan is correct.
- No new dependencies unless user explicitly approves.
- Avoid formatting-only diffs (no reorder imports / refactor unrelated code).
- You shall complete self-verification on your own and keep revising until the check passes without errors.
- At the end of each output, you shall provide the user with recommendations for the next work plan in light of the current work progress/status.
- If the user's requirements violate the above constraints, you shall remind the user and may temporarily bypass the constraints only after obtaining the user's permission.
- If the user's request is inappropriate, you shall remind the user.
- After each change: provide
  1) change summary (3-6 bullets)
  2) compatibility checklist
  3) minimal self-check commands (PowerShell or python -c)

## Repo context
- OS: Windows + PowerShell
- Python: conda env (fgs)
- Typical checks: `python -c "import py_compile; ..."`
