# What went wrong

Every bug, leakage incident, simulator shortcut or failed experiment goes here.
That file is interview material: it shows how problems are found and fixed.

## Template

### Incident
_What happened, one line._

- **How detected:** _test name, guard, manual check, ..._
- **Effect:** _what wrong result it could have produced._
- **Fix:** _what was changed and where._
- **Test added:** _which test now guards against regression._

---

### Incident: prompt files missing from `prompts/`
- **How detected:** `git status` before the Part 01 commit showed only `00_START_HERE.md` in `prompts/`; the `part_*.md` build files were absent.
- **Effect:** the repo would have shipped without its build log/instructions.
- **Fix:** re-copied with `Get-ChildItem -Filter "part_*.md" | Copy-Item` (the original `Copy-Item -LiteralPath ... part_*.md` does not expand wildcards).
- **Test added:** eyeball check of `prompts/` listing before commit (build-process issue, no unit test).

---