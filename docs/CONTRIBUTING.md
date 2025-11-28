# Project Contribution Guidelines

This guide covers contribution workflow, code standards, and development practices for the `biomapper2` project.

## Development Workflow
- Create feature branches from `main`
- Submit pull requests back to `main` (no direct commits to `main`)
- All PRs require passing tests before merge
- Once merged, Auto-sync to arpanauts via GitHub Action

## Before Submitting a PR
Run all code quality checks locally:
```bash
# Check everything (ruff, black, pyright, pytests)
./scripts/check.sh

# Or auto-fix formatting/linting issues first
./scripts/fix.sh
./scripts/check.sh
```

**Note:** CI runs the same checks as `check.sh` automatically on every PR. All checks must pass before merging.

## Commit Messages
We use [Conventional Commits](https://www.conventionalcommits.org/):

| Type      | Description                             | Example                                               |
|-----------|-----------------------------------------|-------------------------------------------------------|
| feat:     | New features                            | feat: add protein annotation support                  |
| fix:      | Bug fixes                               | fix: resolve DataFrame corruption in Step 3           |
| docs:     | Documentation only                      | docs: update installation instructions                |
| test:     | Test additions/changes                  | test: add unit tests for harmonization pipeline       |
| refactor: | Code restructuring (no behavior change) | refactor: extract validation logic to separate module |
| chore:    | Dependency updates, tooling             | chore: update pandas to 2.2.0                         |

## Code Style
We use automated tools to maintain code quality (more details [below](#code-quality-tools-explained)):
- **Black** - Code formatting
- **Ruff** - Linting and import sorting
- **Pyright** - Type checking

### Writing Code
- Use type hints for function signatures
- Add docstrings for public functions and classes
- Follow patterns established in existing modules

Example:
```python
def harmonize_phenotype(raw_data: pd.DataFrame, schema: str) -> pd.DataFrame:
    """
    Harmonize phenotype data to target schema.
    
    Args:
        raw_data: Input DataFrame with raw phenotype measurements
        schema: Target harmonization schema name
        
    Returns:
        Harmonized DataFrame with standardized columns
    """
    ...
```

### Code Quality Tools Explained

We use three complementary tools to maintain code quality:

**Black (Formatter)**
- **What it does:** Changes how code *looks* - whitespace, line breaks, quotes
- **Example:** Reformats `x=1+2` → `x = 1 + 2`
- **Key point:** Purely cosmetic, doesn't change behavior. Very opinionated with little configuration.

**Ruff (Linter)**
- **What it does:** Checks for *code quality issues* and *potential bugs*
- **Examples:**
  - Unused imports: `import sys` but never using it
  - Unused variables: `x = 5` but never using x
  - Bad patterns: using mutable default arguments `def foo(x=[])`
  - Code smells: overly complex functions
  - Style violations: naming conventions, line too long
- **Key point:** Can auto-fix many issues (remove unused imports, sort imports, etc.)

**Pyright (Type Checker)**
- **What it does:** Checks if your types make sense
- **Example:** Catches `def foo(x: int)` being called with `foo("string")`
- **Key point:** Catches type mismatches, None errors, wrong return types. Doesn't change code, just reports errors.

**Summary**
- **Black**: Makes it pretty
- **Ruff**: Makes it clean and catches mistakes  
- **Pyright**: Makes sure types are correct

Ruff and Black have some overlap in formatting (like line length), but they're configured to work together - Ruff respects Black's formatting choices.

All three catch different categories of issues, so running all of them gives comprehensive code quality coverage.

You can run all three checks with `./scripts/check.sh`, and/or configure your IDE to run each tool automatically as you code (see IDE Setup section below).

### IDE Setup (Optional)
For real-time checking in your IDE, install these extensions/plugins:
- **Ruff** - For inline linting warnings and auto-fixes
- **Black** - For automatic formatting on save (VS Code users: install extension from ms-python)
- **Pyright** (optional) - For real-time type checking

With these installed, many issues will be fixed automatically as you code, and you'll see warnings for issues that need manual attention. Exact installation steps vary by IDE.

## Project Tracking

* Use GitHub Issues for task tracking
* Move cards across Kanban board:
  * **Backlog** - Identified tasks
  * **Ready** - Tasks that are ready to be worked on
  * **In progress** - Tasks that are actively being worked on
  * **In review** - Tasks that are complete and pending PR
  * **Done** - Tasks that have had the relevant PR merged
* Link PRs to issues: `Closes #42` in PR description
* Label issues: bug, enhancement, documentation, planning

## Dependency Management
### Adding New Packages
When adding dependencies, always commit both files:
```bash
uv add <package-name>
git add pyproject.toml uv.lock
git commit -m "chore: add <package-name> dependency"
```

## Working with Data Files
By default, the `.gitignore` excludes most data files to keep the repository lightweight and avoid accidentally committing data. If you need to track specific data files (like examples or test fixtures), you can override the ignore patterns:
```bash
# In .gitignore, add an exception:
!data/examples/*.csv
!data/fixtures/test_data.json
```

The `!` prefix tells git to track these files even though the parent pattern would ignore them.