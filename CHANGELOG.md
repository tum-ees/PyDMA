# Change Log

All notable changes to PyDMA are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [2.1.0] - 2026-09-02

### Changed

* Python 3.10 and 3.11 are supported again; `requires-python` drops from 3.12
  to 3.10. The entire 3.12 requirement came down to the two PEP 695 `type`
  aliases in `pydma.utils.roi`, which now use `typing.TypeAlias` and parse on
  every supported interpreter. Runtime behavior is unchanged.
* The pinned-minimum CI leg runs on Python 3.10, and black and mypy target
  3.10. On 3.10 the resolver serves at most `numpy` 2.2 / `scipy` 1.15, since
  newer releases of both dropped 3.10 wheels.

## [2.0.0] - 2026-09-01

### Added

* Fitted series-resistance correction for the full-cell reconstruction (default
  off): `DMAConfig.allow_resistance_offset` with `pocv_current_a`,
  `resistance_offset_limit_ohm_ah`, and `allow_negative_resistance_offset`.
  Parameter slot 9 (`r_offset_ohm`, in Ohm) lifts the model OCV by
  `sign(direction) * R * |I|`. DVA and ICA never read the slot; a constant would
  drop out of their derivatives anyway. With the flag off the slot is pinned to
  zero, and every stored 8-parameter vector expands to the same reconstruction as
  before.

* `apply_aging` in `pydma.utils.balancing`: derives an aged electrode balancing at
  fixed `c_max` by scaling `eps_s` with `1 - LAM`. The former docstring recipe
  cancelled LAM out of `Q_n` and is documented as such.

* `FittedParams.from_dict`, `DMAResult.config_snapshot` for fit provenance, and
  the accepted-run parameter scatter (`param_std`) on `DMAResult`.

* `csv_kwargs` passthrough in the loaders (for example `{"sep": ";", "decimal":
  ","}`), `cathode_convention` in the OCP-model window functions, `x_is_soc` in
  the comparison plots, `include_geometry` in `pybamm_overrides`.

* Loud validation: blend electrodes must match `use_anode_blend`/`use_cathode_blend`,
  OCP curves must be finite, ROI bounds must be SOC fractions, `DMAConfig` rejects
  unknown attributes and re-validates on assignment, `CellGeometry` rejects
  non-positive fields.

* An expanded regression suite: hand-derived LAM/LLI expectations, direct PAV
  pooling checks, plot smoke tests with an rcParams invariance check, aging-study
  gap discovery, and a resistance-offset acceptance suite.

### Changed

* `popsize` now sets the actual differential-evolution population size regardless
  of pinned parameter slots; the effective population no longer shrinks as
  features are disabled.

* `lfp_preset` selects the intended split OCV ROI (0-15 % and 85-100 % SOC)
  instead of the full range.

* `load_aging_study` discovers every present check-up by listing the directory and
  warns about missing indices instead of stopping at the first gap.

* Objective failures raise or warn once instead of silently returning the penalty
  value for every exception; an empty ROI is a configuration error and raises.

* Plot functions no longer mutate global `plt.rcParams`; styles are scoped per
  call.

* Electrode and blend interpolation clamps at the support edges instead of
  returning 0.0 outside the support.

* `derive_balancing_from_result` defaults to `on_out_of_range="raise"` and reports
  clipping explicitly.

* `fit_ocv`/`fit_dva`/`fit_ica`/`combined_objective` take ROI and flag arguments
  keyword-only; the inert `inhom_points` parameter is gone. The three fit terms
  share precomputed electrode potentials with bit-identical results.

### Fixed

* Type checking passes across matplotlib releases. matplotlib 3.11 types the
  `rc_context` mapping by the literal set of rcParams names it ships with, which
  a plain string-keyed dict does not satisfy, so the plotting module applies its
  style through a single wrapper. Plot output is unchanged.

* `voltage_anchored_windows` inverts only the strictly monotonic part of the
  reconstruction, keeping interpolation fill artefacts out of the anchored
  windows.

* DVA/ICA edge guards (zero denominators, fewer than two points, NaN inputs),
  `.mat` NaN filtering, direction-symmetric pOCV file matching, and
  `calculate_mse` returning `inf` for an empty mask.

### Removed

* Dead API: `calculate_full_cell_ocv`, `create_optimizer_from_config`,
  `run_single_fast`, `SiliconCurveParams`, `calculate_inhomogeneity_for_electrode`,
  `DMAConfig.get_bounds`, `DMAConfig.calculate_roi_bounds`, and the
  never-populated `is_cyclic`/`fit_reverse` fields, `get_potential_at_scaled_soc`.

## [1.1.2] - 2026-08-04

### Added

* `tests/test_strict_sto.py` pins the plateau-collapse contract: the plateau
  configuration that used to lose voltage support, the eight-ulp level pooling
  in both curve directions, the inward-only boundary shift, the input and
  post-condition exceptions, and a bit-for-bit comparison against the previous
  arithmetic wherever the plateau levels are well separated. The file carries a
  frozen copy of the pre-1.1.2 implementation as its contrast reference, so the
  input class that used to be damaged stays documented in executable form.

* The `pchip_resample_for_pybamm` docstring now states that its deduplication is
  the intended handling for the genuine plateaus of a raw PAV curve, and that a
  collapsed curve reaches it tie-free, so a residual tie there indicates a defect
  in `_collapse_plateaus` rather than a plateau to be squashed.

### Changed

* `_collapse_plateaus` states its contract with real exceptions instead of bare
  `assert`, so the guarantees survive `python -O`. Unusable input raises
  `ValueError`: a voltage and capacity pair of unequal length, a non-finite
  voltage, capacity or `eps` value, a capacity curve that after the monotone
  snap is constant, and finite samples whose overall range overflows the
  floating-point span. The finiteness checks matter because every comparison
  against NaN is false, so a NaN used to pass the degeneracy guard, the repair
  sweep and the post-conditions untouched and left the caller with a silently
  NaN-poisoned curve. A collapsed curve that is non-finite, not strictly
  monotone, or that does not preserve both range endpoints exactly raises
  `RuntimeError`, because these properties now hold by construction and a
  violation would mean a defect in the function rather than an unusable input.
  Error messages identify the offending samples and values. Monotonicity and
  endpoint errors additionally report both capacities and voltages.

* Only the opt-in collapse path changes. `collapse_plateaus` still defaults to
  `False`, `pchip_resample_for_pybamm` is numerically untouched, and silicon OCP
  tables exported through the default raw-PAV path reproduce bit-for-bit, checked
  against the committed M35A lithiation and delithiation tables. Callers that
  pass `collapse_plateaus=True` receive intentionally corrected values wherever
  the quarter-gap bound now applies. On M35A-like data that correction reaches
  about 1e-5 in capacity and stays below 0.005 mV after the 1001-point PCHIP
  resample. Where the plateau levels are well separated the new code reproduces
  the previous arithmetic exactly.

* Pinned the `ruff` development dependency to `>=0.15.21,<0.16`, matching the
  Ruff 0.15 release line the pre-commit hook already used. The previous
  floating floor let a
  fresh environment resolve to the 0.16 rule generation, which reports 33
  findings on the existing code base and so turned the lint gate red
  independently of any change. Those findings need their own review pass and are
  not part of this release.

### Fixed

* `generate_si_curve(collapse_plateaus=True)` no longer loses part of the voltage
  support when two PAV plateau levels sit closer together than the tie-breaking
  shift, which happens next to the saturation boundary at q = 0 and q = 1.
  Previously the two shifts crossed, the monotonicity sweep pushed the samples
  past the capacity range, and the final clip put them back onto the range
  boundary, where they became exact ties. The deduplication in
  `pchip_resample_for_pybamm` then dropped the tied samples together with their
  voltages and truncated the exported curve while every check stayed green. The
  shift is now bounded by a quarter of the distance to the neighbouring plateau
  levels, a plateau sitting on the range boundary is shifted inward only, and the
  final clip is gone, so the collapsed capacity can no longer step outside the
  range of the input curve. This mirrors the fix shipped in the MATLAB
  DegradationModeAnalysis framework 2.1.0.

* Plateau levels within eight floating-point ulps of each other are pooled into a
  single plateau instead of being shifted apart, because at that distance a
  quarter of the gap is no longer representable and the shifted endpoint rounds
  straight back onto its own level. The merged plateau keeps the first and the
  last sample of the whole group, so its voltage extent is preserved. When the
  whole capacity range is only a few ulps wide and the pooling therefore merges
  the entire curve into one plateau, the collapse reports the two range edges
  directly, so both edges keep their exact value instead of one of them being
  shifted off it.

* The ulp arithmetic inside `_collapse_plateaus` is measured on absolute
  magnitudes. `numpy.spacing` returns a negative step for a negative argument,
  and the internal sign flip that lets a falling curve share the rising code path
  makes every working value negative there, so measuring on the raw value pointed
  the level pooling and the repair sweep the wrong way for every discharge curve.

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
