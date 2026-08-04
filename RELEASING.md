# Releasing PyDMA

Maintainer checklist for cutting a new PyDMA release. Run top-to-bottom;
do not skip steps because "it was fine last time" — the gates are here
because each one has caught a bug in a prior release.

Replace `X.Y.Z` with the new version everywhere below.

---

## 1. Prepare the release branch

Work on a `release/vX.Y.Z` branch off `main` (or off the previous
release branch if you are picking up partial work). Land all
scope-bearing commits first; the steps below are the final polish.

```bash
git switch -c release/vX.Y.Z main
```

## 2. Bump the version and citation metadata

Single source of truth: [src/pydma/_version.py](src/pydma/_version.py).
`pyproject.toml` reads it dynamically via `[tool.setuptools.dynamic]`.

```python
# src/pydma/_version.py
__version__ = "X.Y.Z"
```

Update both release fields in [CITATION.cff](CITATION.cff) at the same time:

```yaml
version: X.Y.Z
date-released: YYYY-MM-DD
```

## 3. Update CHANGELOG.md

Add a dated entry at the top, mirroring the existing format (`### Added`
/ `### Changed` / `### Fixed` / `### Removed`):

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Changed
* …

### Added
* …
```

## 4. Update README.md highlights

Add an `**X.Y.Z highlights:**` block at the top of the Release Notes
section in [README.md](README.md), 3-5 bullets max. Lead with
user-visible behavior changes; leave internal cleanup (mypy, typing,
docs) for one short trailing bullet.

## 5. Run all gates locally

Every gate must pass before you push. If any fails, fix it on the
release branch.

```bash
# Fast unit suite
python -m pytest --override-ini="addopts=" -q -m "not scientific"

# Seeded scientific regression
python -m pytest --override-ini="addopts=" -q -m scientific

# Type check
python -m mypy src/pydma

# Lint
python -m ruff check src tests doc

# Formatting (same pins CI enforces; a reformat here is a real gate failure)
python -m black --check src tests doc
python -m isort --check-only src tests doc

# Tutorial sanity-check (manual): run BOTH tutorial notebooks end-to-end on
# the release branch. Treat the run as a gate, not as a source of commits.
# The fits use differential evolution without a seed, so the numbers move
# substantially between runs on identical code. Measured across two runs of
# the same commit: joint-fit RMSE from 3.9 to 8.3 mV, blend gamma_Si from
# 0.224 to 0.234. Afterwards revert the stochastic notebook outputs and
# notebooks/dma_results_P45B_23.json, then commit only what you actually
# meant to change, such as the version stamps and any deliberate source edits.
#
#   notebooks/getting_started.ipynb
#     Confirm the joint fits still land in the single-digit mV band and that
#     blend gamma_Si is still about 0.22.
#
#   notebooks/pybamm_integration.ipynb
#     Confirm the printed PyDMA version stamp matches the release and that
#     saved output does not contain an absolute workstation path. Also confirm
#     the PASS guard "DFN voltage RMSE within 30 mV tolerance" still
#     holds at the end.
```

Working tree must be clean before the merge commit:

```bash
git status --short    # expect empty output
```

## 6. Hygiene re-checks each cut

These are the items review keeps catching. Verify explicitly:

- [ ] If a `[tool.setuptools.package-data]` section is present in
      `pyproject.toml`, every declared path actually exists on disk
      (the 1.1.0 cut removed a stale `pydma.data.ocps` entry that
      pointed at a non-existent directory; do not let that pattern
      reappear).
- [ ] No `from __future__ import annotations` reintroduced in any file
      (we are on Python ≥ 3.12; the shim is no longer needed).
- [ ] Dependency floors in `pyproject.toml` have wheels for the
      supported Python floor (e.g. `numpy>=1.26.0` is the lowest with
      3.12 wheels).
- [ ] `notebooks/getting_started.ipynb` and any saved-result JSONs are
      either intentionally refreshed *or* reverted before the merge
      commit — do not ship stochastic-rerun noise as "the release."
- [ ] Scientific test still skips cleanly from an unpacked sdist (see
      §8 verification).

## 7. Merge and tag

```bash
git switch main
git merge --no-ff release/vX.Y.Z       # or use the platform's merge UI
git tag -a vX.Y.Z -m "Release X.Y.Z"
git push origin main
git push origin vX.Y.Z
```

The tag goes on the *merge* commit, not on any commit on the release
branch. PyPI does not require the tag, but it makes "what code is on
PyPI version X.Y.Z" trivially answerable.

## 8. Build artifacts from the tagged commit

Always build from a fresh clean checkout of the tag, not from the
release branch with potential local untracked files.

```bash
git clone --depth 1 --branch vX.Y.Z <repo-url> /tmp/pydma-release
cd /tmp/pydma-release
python -m build                          # produces dist/pydma-X.Y.Z.tar.gz + .whl
python -m twine check dist/*             # both must report PASSED
```

Smoke check the sdist-installed behavior:

```bash
python -m venv /tmp/pydma-sdist-venv
/tmp/pydma-sdist-venv/bin/python -m pip install dist/pydma-X.Y.Z.tar.gz
/tmp/pydma-sdist-venv/bin/python -c "import pydma; print(pydma.__version__)"
# Expect: X.Y.Z
```

Verify the scientific test skips cleanly when the unpacked sdist
lacks the data files (this is the contract the module-level
`pytest.mark.skipif` in `tests/test_scientific_regressions.py`
promises; if it breaks, that contract is broken):

```bash
tar -xzf dist/pydma-X.Y.Z.tar.gz -C /tmp
python -m pytest /tmp/pydma-X.Y.Z/tests/test_scientific_regressions.py \
    -m scientific -q -o addopts=""
# Expect: 2 skipped
```

## 9. Upload to TestPyPI, smoke-install

```bash
python -m twine upload --repository testpypi dist/*
```

In a fresh venv, install from TestPyPI and import:

```bash
python -m venv /tmp/pydma-testpypi-venv
/tmp/pydma-testpypi-venv/bin/python -m pip install \
    -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    pydma==X.Y.Z
/tmp/pydma-testpypi-venv/bin/python -c "from pydma import DMAAnalyzer; import pydma; print(pydma.__version__)"
```

## 10. Upload to real PyPI

Only after the TestPyPI smoke test passes:

```bash
python -m twine upload dist/*
```

Confirm in a fresh venv:

```bash
python -m venv /tmp/pydma-pypi-venv
/tmp/pydma-pypi-venv/bin/python -m pip install pydma==X.Y.Z
/tmp/pydma-pypi-venv/bin/python -c "import pydma; print(pydma.__version__)"
```

This is the run that actually exercises the lower-direct dependency
floors against PyPI's resolver.

## 11. Announce / close out

- [ ] Move any remaining release-branch-only commits back into the
      mainline narrative if needed.
- [ ] Delete the release branch (`git branch -d release/vX.Y.Z` and
      `git push origin --delete release/vX.Y.Z`).

---

## Appendix: anti-patterns this checklist exists to prevent

- Pushing a release with a CHANGELOG entry but no matching README
  highlight (or vice versa).
- Bumping `_version.py` and forgetting that `pyproject.toml` reads it
  dynamically — there is no second place to update.
- Building artifacts from the development checkout (picks up untracked
  files, stale `dist/`, or build/ directories).
- Skipping `twine check` and finding out PyPI rejects the README at
  upload time.
- Shipping `from __future__ import annotations` shims that contradict
  the stated Python floor.
- Committing notebook outputs that are pure stochastic-rerun noise
  with no real content change.
