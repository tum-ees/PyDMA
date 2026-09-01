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

A major release additionally leads the README with a
`## Highlights of X.Y.Z` section, because the README is the PyPI project
page and a breaking change belongs above the fold. In that case the
Release Notes entry is a one-line pointer to it instead of a copy, and
the next major cut replaces the top section rather than adding a second
one.

## 5. Run all gates locally

Every gate must pass before you push. If any fails, fix it on the
release branch.

Local results depend on the versions installed in your environment,
while the CI jobs resolve dependencies fresh on every run, so a gate can
pass locally and fail in the pipeline on the same commit (the 2.0.0 cut
hit this: matplotlib 3.11 narrowed the `rc_context` typing while the
local environment still ran 3.10). The pipeline on the merge request is
the authoritative check. To reproduce it locally, run the gates in a
fresh venv installed with `pip install -e ".[dev]"`.

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
# the release branch. The two notebooks behave differently here:
#
#   notebooks/getting_started.ipynb
#     Runs seeded (random_seed=1234, medium preset), so its numbers are
#     reproducible to the printed digit on identical code and versions —
#     a rerun that lands elsewhere is a finding, not noise. Refreshing
#     its outputs together with notebooks/dma_results_P45B_23.json is
#     therefore meaningful; commit both from the same run or neither.
#     Confirm the joint fits land in the single-digit mV band and that
#     blend gamma_Si is still about 0.22.
#
#   notebooks/pybamm_integration.ipynb
#     Unseeded; its fit numbers move between runs on identical code, so
#     refresh its outputs only for a real content change. Confirm the
#     printed PyDMA version stamp matches the release and that saved
#     output does not contain an absolute workstation path. Also confirm
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
      §9 verification).

## 7. Merge and tag

`main` is a protected branch and only accepts merge requests, so merge
through the GitLab UI, then tag the merge commit it created:

```bash
git switch main
git pull --ff-only
git tag -a vX.Y.Z -m "Release X.Y.Z"
git push origin vX.Y.Z
```

The tag goes on the *merge* commit, not on any commit on the release
branch. PyPI does not require the tag, but it makes "what code is on
PyPI version X.Y.Z" trivially answerable.

Tag only after the pipeline on `main` is green. A tag is cheap to place
and expensive to move once anything has consumed it, and the pipeline is
the first consumer that notices a broken one.

## 8. Verify the public mirror

GitLab mirrors this repository to the GitHub mirror
(`github.com/tum-ees/PyDMA`) automatically shortly after each push,
branches and tags included (Settings → Repository → Mirroring).
Everything from here on runs on that mirror: the publish workflow
uploads to PyPI from there, and Zenodo mints the DOI from a Release
published there. The mirror also carries the URLs that PyPI shows on
the project page (Homepage, Repository, Issues all point at GitHub).

Confirm the sync arrived before continuing — `main` and the tag must
resolve to the same commits on both sides:

```bash
git ls-remote https://github.com/tum-ees/PyDMA.git refs/heads/main "refs/tags/vX.Y.Z*"
git rev-parse main "vX.Y.Z^{commit}"
```

Only if the mirror lags or the sync is broken, push by hand:

```bash
git remote add mirror https://github.com/tum-ees/PyDMA.git   # if missing
git push mirror main
git push mirror vX.Y.Z
```

The tag must contain `.github/workflows/publish.yml`: a workflow
triggered by a release runs the file as it exists in the tagged commit,
not the one on the default branch, so a tag without it uploads nothing,
silently.

## 9. Dry run against TestPyPI

Trusted publishing is registered on PyPI and on TestPyPI for this
repository and workflow (`publish.yml`, environments `pypi` and
`testpypi`); no token is involved. On the mirror, run *Actions → publish
→ Run workflow* with target `testpypi`. The workflow builds from `main`,
checks the artifacts with twine, and uploads to TestPyPI. An
`invalid-publisher` error means one of owner, repository, workflow
filename, or environment name differs between the workflow and the
publisher registered on the index.

In a fresh venv, install from TestPyPI and import (on Windows the venv
binaries live under `Scripts\` instead of `bin/`):

```bash
python -m venv /tmp/pydma-testpypi-venv
/tmp/pydma-testpypi-venv/bin/python -m pip install \
    -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    pydma==X.Y.Z
/tmp/pydma-testpypi-venv/bin/python -c "from pydma import DMAAnalyzer; import pydma; print(pydma.__version__)"
```

Verify the scientific test skips cleanly from the sdist that was
actually uploaded (this is the contract the module-level
`pytest.mark.skipif` in `tests/test_scientific_regressions.py`
promises; if it breaks, that contract is broken):

```bash
python -m pip download --no-deps --no-binary :all: \
    -i https://test.pypi.org/simple/ pydma==X.Y.Z -d /tmp/pydma-sdist
tar -xzf /tmp/pydma-sdist/pydma-X.Y.Z.tar.gz -C /tmp/pydma-sdist
python -m pytest /tmp/pydma-sdist/pydma-X.Y.Z/tests/test_scientific_regressions.py \
    -m scientific -q -o addopts=""
# Expect: 2 skipped
```

## 10. Publish the Release: PyPI and Zenodo in one step

Only after the TestPyPI dry run passes. On the mirror: *Releases → Draft
a new release*, pick tag `vX.Y.Z`, title `vX.Y.Z`, body = the CHANGELOG
entry. Publishing triggers both consumers of the release event at once:

- the publish workflow uploads to PyPI with PEP 740 attestations,
  refusing first if the tag disagrees with `_version.py`;
- Zenodo archives the tag within seconds.

Verify all of it:

- [ ] The `publish` run in the Actions tab is green.
- [ ] `pip install pydma==X.Y.Z` works in a fresh venv and the import
      prints the right version. The simple index that pip reads lags the
      JSON API by a few minutes, so a "no matching distribution" right
      after the upload is propagation, not failure. This install is also
      the run that exercises the lower-direct dependency floors against
      PyPI's resolver.
- [ ] A new version record exists under the concept DOI
      [10.5281/zenodo.21346639](https://doi.org/10.5281/zenodo.21346639),
      and its version field reads `vX.Y.Z`.
- [ ] `CITATION.cff` still carries the *concept* DOI, not the new
      per-version DOI. The concept DOI always resolves to the newest
      version, which is why it does not change from release to release.

New files can be added to a PyPI release for 14 days after publication;
after that PyPI rejects them and the only correction left is yanking.

## 11. Fallback: manual upload with twine

The API token stays valid alongside trusted publishing. If the workflow
path is unavailable, build from a fresh clean checkout of the tag, not
from the release branch with potential local untracked files:

```bash
git clone --depth 1 --branch vX.Y.Z <repo-url> /tmp/pydma-release
cd /tmp/pydma-release
python -m build                          # produces dist/pydma-X.Y.Z.tar.gz + .whl
python -m twine check dist/*             # both must report PASSED
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*            # only after the TestPyPI smoke test
```

The verifications from §9 and §10 apply unchanged. A manual upload
cannot carry attestations; PyPI accepts those only from a
trusted-publisher identity.

## 12. Announce / close out

- [ ] Move any remaining release-branch-only commits back into the
      mainline narrative if needed.
- [ ] GitLab deletes the source branch when the merge request merges;
      delete the local one (`git branch -d release/vX.Y.Z`) and prune
      the stale remote-tracking refs (`git fetch --prune`).

---

## Appendix: anti-patterns this checklist exists to prevent

- Pushing a release with a CHANGELOG entry but no matching README
  highlight (or vice versa).
- Bumping `_version.py` and forgetting that `pyproject.toml` reads it
  dynamically — there is no second place to update.
- Building artifacts from the development checkout (picks up untracked
  files, stale `dist/`, or build/ directories).
- Tagging a commit that predates `publish.yml`: the release event runs
  the workflow file from the tagged commit, so nothing uploads and
  nothing errors.
- Skipping `twine check` and finding out PyPI rejects the README at
  upload time.
- Shipping `from __future__ import annotations` shims that contradict
  the stated Python floor.
- Committing notebook outputs that are pure stochastic-rerun noise
  with no real content change.
- Uploading to PyPI and stopping there: without the mirror push and the
  GitHub Release, the version has no Zenodo record, so the DOI in the
  citation metadata keeps resolving to the previous release.
