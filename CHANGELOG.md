# Change Log

All notable changes to PyDMA are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [1.1.1] - 2026-07-13

### Added

* Added `CITATION.cff` metadata for software citation and Zenodo ingestion.
* Added GitLab and GitHub CI definitions for formatting, linting, typing,
  tests, scientific regression checks, and package builds.

### Changed

* Applied the repository's pinned Black and isort formatting consistently.
* Aligned development dependencies and pre-commit hooks with the documented
  release gates.
* Clarified the bundled P45B test-data citation and removed
  workstation-specific notebook output.
* Updated type annotations for current NumPy and Matplotlib typing without
  changing runtime behavior.

### Fixed

* Fixed platform-dependent charge/discharge CSV selection in aging-study
  directories.

## [1.1.0] - 2026-05-23

### Changed

* **Dropped Python 3.9, 3.10, 3.11 support.** New minimum is Python 3.12.
  Removed the `from __future__ import annotations` workarounds in
  `silicon/strict_sto.py`, `utils/balancing.py`, `utils/roi.py`,
  `preprocessing/loader.py`, `tests/test_balancing.py` and
  `doc/create_gif.py`, which were there only to make Python 3.10+
  `|`-union annotations parse on 3.9.
  `pyproject.toml` updated: `requires-python = ">=3.12"`,
  `[tool.black] target-version = ["py312"]`, `[tool.mypy] python_version
  = "3.12"`, classifier list trimmed to 3.12 only.

* **Bumped runtime dependency floors** to the lowest releases with
  Python 3.12 wheels: `numpy>=1.26.0`, `scipy>=1.11.4`,
  `pandas>=2.1.1`, `matplotlib>=3.8.0`, `statsmodels>=0.14.0`. The
  previous floors (numpy 1.20, scipy 1.7, pandas 1.3, matplotlib 3.4,
  statsmodels 0.13) were pre-3.12 and would fail to install on a fresh
  3.12 environment because they pulled `numpy==1.20.0`, which depends
  on the removed `distutils` module.

* **Typing modernization (PEP 585 / 604 / 695)**: `Tuple/List/Dict/...`
  → lowercase generics; `Optional[X]` → `X | None`; `Union[X, Y]` →
  `X | Y`; `TypeAlias` in `utils/roi.py` → `type` statement.
  Mechanical rewrite via `pyupgrade --py312-plus` followed by
  `ruff --select F401 --fix` for the now-unused `typing` imports. Zero
  behaviour change; all 37 tests pass; `notebooks/getting_started.ipynb`
  re-executes cleanly.

### Added

* `pydma.silicon.strict_sto.pchip_resample_for_pybamm`: PCHIP
  shape-preserving resample of the raw PAV-filtered silicon OCP onto a
  uniform sto grid (default 1001 points) with an optional endpoint-V
  snap to the silicon-saturation V. Produces a smooth strictly-monotone
  CSV that PyBaMM's CasADi/IDAS interpolant can consume without the
  ~50 mV V_cell jump that the densely-preserved curve creates at the
  silicon saturation knee during a discharge. Can also be used as a
  `DMAAnalyzer` refit input when `DMAConfig(smoothing_points=1)` is set
  (the default 30 over-smooths the already-PCHIP-smoothed curve and
  basin-escapes). Replaces the previous `strict_sto_eps_spread` which
  preserved V-density (good for PyDMA refit) but left the PyBaMM-side
  knee unmitigated.

* `DMAConfig.random_seed` for deterministic optimizer runs (promoted
  from a hardcoded `property -> None` to a real dataclass field;
  default `None` preserves prior nondeterministic behavior), and an
  opt-in `pytest -m scientific` golden regression suite that runs a
  full end-to-end DMA fit against pinned RMSE / fitted-parameter /
  degradation-mode numbers (`tests/test_scientific_regressions.py`,
  `tests/golden/p45b_serial23_entry01_cu1_nonblend.json`). Catches
  silent behavior shifts the unit tests miss.

## [1.0.2] - 2026-05-12

### Added

* New `pydma.utils.balancing` module that maps a PyDMA voltage-anchored
  fit and the cell geometry onto simulator-agnostic
  `c_max` / `c_init(SoC)` values. Public exports at the package
  top-level: `CellGeometry`, `ElectrodeBalancing`, `derive_balancing`,
  `derive_balancing_from_result`.
  `ElectrodeBalancing.pybamm_overrides(soc)` returns a dict keyed by
  PyBaMM's exact parameter-name strings for direct use with
  `pybamm.ParameterValues.update(...)`. The math is universal
  full-cell electrochemistry; only that one method is PyBaMM-flavoured.
* `tests/test_balancing.py` regression suite covering charge-balance
  derivation, PyBaMM `x_p` cathode-window convention, voltage-anchored
  utilization, partial-cutoff validation, and aged check-up usage with
  `eps_s_* * (1 - LAM_*)` scaling.
* New `notebooks/pybamm_integration.ipynb` end-to-end tutorial: takes a
  PyDMA fit on the Molicel INR21700-P45B BoL pseudo-OCV, derives
  `c_max` / `c_init` via `derive_balancing_from_result`, builds a
  PyBaMM `ParameterValues`, and verifies the result with a C/500 DFN
  charge round-trip. All material-property values are sourced from
  Frank et al. (2025), Table III
  (DOI 10.1149/1945-7111/adc03c); `Chen2020` is used only as a
  generic public Li-ion fallback base for slots Frank does not
  document.
* `notebooks/parameter_data/frank2025_p45b_table_iii.json`: 24 Frank
  Table III constants stored as `{value, unit, source}` per entry,
  acting as the single source of truth consumed by the new
  notebook.

### Changed

* `notebooks/getting_started.ipynb` is now purely a DMA-analysis
  tutorial. The inline PyBaMM bridge has been moved out into the
  dedicated `pybamm_integration.ipynb` and the remaining sections
  renumbered.
* `pyproject.toml` now reads the package version dynamically from
  `src/pydma/_version.py` via `[tool.setuptools.dynamic]` instead of
  carrying a literal version string. Single source of truth for the
  package version.

## [1.0.1] - 2026-04-28

### Fixed

* Stoichiometry export can now be anchored to the fitted reconstructed cell
  voltage at the requested voltage limits via
  `DMAResult.voltage_anchored_windows(...)`. This avoids exporting raw
  internal fit-window endpoints that do not necessarily correspond to the
  measured pseudo-OCV cutoffs. For inhomogeneous fits, the anchored values are
  the central/nominal stoichiometries of the fitted trajectory.
* Silicon OCP plateau collapse now returns strictly monotone output for
  `generate_si_curve(monotone_filter=True)`, making the filtered curve safe for
  downstream spline interpolation without changing fitting results.

### Added

* `BlendElectrode.get_component_stoichiometries(...)` and
  `BlendElectrode.get_component_stoichiometry_window(...)` for mapping a blend
  coordinate to per-phase graphite/silicon stoichiometries.
* `FittedParams.sto_window_an_per_phase(...)` for raw per-phase inspection when
  needed.
* Voltage-anchored composite export helpers for downstream workflows and an
  updated `getting_started.ipynb` demonstration of fitted-reconstruction
  voltage anchoring plus anchored Gr/Si phase windows.

## [1.0.0] - 2026-04-13

### Added

* Inhomogeneity offset for anode and cathode (`DMAConfig.inhom_anode_offset`,
  `DMAConfig.inhom_cathode_offset`, default `0.0`, validated `[0, 1]`). A
  positive offset allows a fraction of the maximum inhomogeneity spread to
  be present already at SOC = 0 instead of starting from zero. Setting the
  offset to `1.0` reproduces SOC-independent inhomogeneity, which is
  analogous to earlier degradation mode analysis frameworks in literature.
  Matches MATLAB's new `inhomOffsetFraction` argument
  (`calculate_inhomogeneity.m`).
* `DMAAnalyzer.analyze_aging_study(path, ...)` convenience API that accepts
  a directory or single `.mat` file, loads it using the configured
  `direction`, and runs every CU.
* Top-level `load_aging_study` export and support for single-file multi-CU
  `.mat` payloads in the loader.
* Isotonic-regression-based silicon OCP filtering in `generate_si_curve`,
  producing strictly monotonic curves while keeping the maximum amount of
  information.
* Regression tests pinning MATLAB-parity invariants (q0, fitted-bounds,
  degradation-mode delegation, inhomogeneity offset formula, loader CU
  handling, reset-state completeness, `FittedParams` `None` handling).

### Changed

* **Numerical:** `q0` now matches MATLAB (span of the normalized SOC axis,
  ≈ 1.0) instead of the raw Ah span. Because the DVA and ICA cost
  contributions scale as `q0²`, this may produce **small numerical
  differences compared with older PyDMA versions when `weight_dva` and/or
  `weight_ica` are non-zero**. In return, fits are now cell-size independent
  and consistent with the MATLAB-tuned `weight_dva` / `weight_ica` defaults.
  OCV-only fits (`weight_dva = 0`, `weight_ica = 0`) are unaffected.
* **Breaking:** `compare_with_reference` now delegates to
  `calculate_degradation_modes`, so blend LAMs
  (`lam_anode_blend1`/`2`, `lam_cathode_blend1`/`2`) are populated rather
  than silently zero. Constructor keyword arguments for `DegradationModes`
  renamed `lam_an` / `lam_ca` → `lam_anode` / `lam_cathode` (the short
  names remain available as read-only property aliases).
* `DMAConfig.algorithm` is now actually consumed by the optimizer, and
  `DMAConfig.get_initial_guess()` now includes blend-weight initial values.
* Aging-study runs now populate real `fit_ocv_mse` / `fit_dva_mse` /
  `fit_ica_mse` and the `is_accepted` / `status` / `algorithm` metadata
  on every `DMAResult` instead of placeholder values.
* Aging-study loader honors the configured `direction` when choosing
  per-CU folders.

### Fixed

* Inhomogeneity out-of-range clamping now uses
  `np.interp(..., left=voltage[0], right=voltage[-1])`, matching MATLAB
  `griddedInterpolant(..., 'linear', 'nearest')`. The previous code
  clamped both OOB sides to `voltage[-1]`, a latent mismatch that affected
  fits even when `inhom_offset = 0`.
* `compare_with_reference` now falls back to
  `self.reference_data.reference_capacity`, and its capacity-loss guard
  protects against division by zero (`reference_capacity == 0`).
* `FittedParams` type annotations and `to_array()` correctly handle `None`
  for disabled blend/inhomogeneity parameters and always return a
  `float64` numpy array.
* `DMAAnalyzer.reset_state` now also clears `_capacity_history` and the
  normalized-SOC warning flag, so repeated aging studies start clean.
* Loader now handles nested `.mat` structs (`mat_struct`) and single-file
  multi-CU `.mat` payloads, and falls back to direction-based folder
  matching when no explicit CU markers are present.
* Reoriented-OCV warning is now one-shot and only fires when
  auto-correction implies a direction opposite to `config.direction`.
* `DMAConfig` now validates blend initial guesses against their upper
  bounds and rejects out-of-range inhomogeneity offsets at construction
  time.

### Removed

* Dead `DMAConfig` fields and the unused internal `direction` threading
  in the loader's CU matcher.

## [0.1.0] - 2026-02-11

* Initial public release on PyPI.
* Core workflow for mode identification (OCV / DVA / ICA fitting, blend
  electrodes, inhomogeneity modeling, silicon OCP generation).
* Jupyter-based getting-started notebook and documentation.
