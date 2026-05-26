# Release Checklist

This checklist must be followed and fully checked off by the maintainer before building and publishing any new release of CortexGit to PyPI.

---

## Pre-Release Verification
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code is formatted with black: `black src/ tests/`
- [ ] Code is linted with ruff: `ruff check src/ tests/`
- [ ] README.md is accurate and up-to-date
- [ ] CHANGELOG.md is updated with release notes and date

## Dependency & Metadata Consistency
- [ ] Version updated in exactly 3 places to the release version:
  - `setup.py`
  - `pyproject.toml`
  - `src/cortexgit/__init__.py`
- [ ] `setup.py` and `pyproject.toml` dependencies and metadata are fully matched and synchronized

## Package Build & Quality Check
- [ ] Old `build/` and `dist/` folders are deleted to prevent stale archives
- [ ] `python -m build` runs cleanly and generates `.tar.gz` and `.whl` files
- [ ] `twine check dist/*` passes with no rendering or description errors

## Environment & Integration Testing
- [ ] Package is tested and verified in a clean virtual environment (`test_env`)
- [ ] Basic imports and functionalities work properly in isolation

## Git Release & Publishing
- [ ] All code changes, documentation updates, and version bumps are committed to the `main` branch
- [ ] Git tag created matching the version (e.g. `git tag v0.1.0`)
- [ ] Git tags pushed to repository: `git push origin --tags`
- [ ] Run `twine upload dist/*` to publish the verified release package to PyPI
