# Change Log

All notable changes to PyDMA are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

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
