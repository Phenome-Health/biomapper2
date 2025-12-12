# PR Preparation Checklist

Before creating a PR, run through this checklist to catch common issues.

## Instructions

Please check the following items for this codebase. Report any issues found and suggest fixes.

## 1. Branch Status
- Is the branch up to date with main? Run: `git fetch origin main && git log --oneline HEAD..origin/main`
- If behind, merge main: `git merge origin/main`
- Are there any merge conflicts?

## 2. Dependencies (pyproject.toml)
- Check for duplicate entries in `[dependency-groups]` section
- If dependencies were added, verify both `pyproject.toml` and `uv.lock` are staged

## 3. Code Patterns (for biomapper2)
For any annotator implementations, verify:
- Annotators return **raw API field names** (not normalized vocab names like `pubchem.compound`)
- No ID cleaning in annotators (Normalizer handles this - e.g., don't strip "RM" prefix)
- All available API fields are captured (not just a subset)

## 4. Quality Checks
Run: `./scripts/check.sh`
- Ruff linting passes
- Black formatting passes
- Pyright type checking passes (0 errors)
- All tests pass

## 5. Tests
- New code has test coverage
- Integration tests are marked with `@pytest.mark.integration`

## 6. Documentation
- CLAUDE.md is up to date (if patterns changed)
- Demo notebooks reflect current API behavior

---

After reviewing, provide a summary of:
1. Issues found (if any)
2. Suggested fixes
3. Whether the PR is ready to submit
