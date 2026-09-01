"""Optimizer wrapper for DMA fitting.

This module wraps scipy.optimize.differential_evolution to provide:
- Multi-run optimization strategy (reqAccepted runs with RMSE < threshold)
- Speed presets (fast, medium, slow) with different population sizes
- Callback support for progress tracking
- Result aggregation across multiple runs

The high population size (500) is critical for good results, as it helps
the evolutionary algorithm explore the complex, multi-modal parameter space.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, differential_evolution

from pydma.utils.dma_config import DMAConfig

# The one UserWarning scipy raises from within differential_evolution that
# describes an expected outcome of this wrapper's own settings. Everything else
# scipy has to say, the caller gets to see.
_SCIPY_DE_WARNING = "differential_evolution: the 'workers' keyword"


@dataclass
class OptimizationRun:
    """Result of a single optimization run."""

    params: NDArray[np.floating]
    """Optimized parameters."""

    cost: float
    """Final cost function value."""

    rmse: float
    """RMSE of the fit."""

    success: bool
    """Whether optimization converged."""

    n_iterations: int
    """Number of iterations performed."""

    n_function_evals: int
    """Number of function evaluations."""


@dataclass
class MultiRunResult:
    """Result of multi-run optimization."""

    best_params: NDArray[np.floating]
    """Parameters of the accepted run with the lowest OCV RMSE.

    The selection is made on ``OptimizationRun.rmse`` (OCV RMSE in Volts from
    the optimizer's ``rmse_fn``), not on the weighted cost. When no run meets
    the threshold, the rejected run with the lowest RMSE is reported instead.
    """

    best_cost: float
    """Best cost value."""

    best_rmse: float
    """Best RMSE value."""

    accepted_runs: list[OptimizationRun] = field(default_factory=list)
    """List of accepted runs (RMSE < threshold)."""

    rejected_runs: list[OptimizationRun] = field(default_factory=list)
    """List of rejected runs (RMSE >= threshold)."""

    std_params: NDArray[np.floating] | None = None
    """Standard deviation of parameters over the accepted runs, ``None`` when
    no run was accepted. A single accepted run makes it all zeros."""

    @property
    def n_accepted(self) -> int:
        """Number of accepted runs."""
        return len(self.accepted_runs)

    @property
    def n_rejected(self) -> int:
        """Number of rejected runs."""
        return len(self.rejected_runs)

    @property
    def n_total(self) -> int:
        """Total number of runs."""
        return self.n_accepted + self.n_rejected

    @property
    def acceptance_rate(self) -> float:
        """Fraction of runs that were accepted."""
        if self.n_total == 0:
            return 0.0
        return self.n_accepted / self.n_total

    @property
    def best_is_accepted(self) -> bool:
        """Whether the selected best run met the RMSE threshold."""
        return self.n_accepted > 0


class DMAOptimizer:
    """Optimizer for DMA parameter fitting.

    This class wraps scipy.optimize.differential_evolution to provide
    multi-run optimization with RMSE threshold acceptance.

    Parameters
    ----------
    config : DMAConfig
        Configuration object with optimization settings
    objective : Callable
        Objective function to minimize, should accept params as first argument
    bounds : list[tuple[float, float]]
        Parameter bounds as list of (min, max) tuples
    callback : Callable, optional
        Callback function called after each iteration

    Attributes
    ----------
    config : DMAConfig
        Configuration object
    objective : Callable
        Objective function
    bounds : list[tuple[float, float]]
        Parameter bounds

    Examples
    --------
    >>> from pydma.core.optimizer import DMAOptimizer
    >>> from pydma.utils.dma_config import DMAConfig
    >>>
    >>> config = DMAConfig(speed_preset='medium')
    >>> optimizer = DMAOptimizer(config, my_objective, bounds)
    >>> result = optimizer.run()
    """

    def __init__(
        self,
        config: DMAConfig,
        objective: Callable[[NDArray[np.floating]], float],
        bounds: list[tuple[float, float]],
        callback: Callable[[NDArray[np.floating], float], None] | None = None,
        rmse_fn: Callable[[NDArray[np.floating]], float] | None = None,
    ):
        self.config = config
        self.objective = objective
        self.bounds = bounds
        self.callback = callback
        self.rmse_fn = rmse_fn

        # Get solver options from config
        self._solver_opts = config.get_solver_options()

    def _run_single(
        self,
        seed: int | None = None,
        **kwargs: Any,
    ) -> OptimizationRun:
        """Run a single optimization.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        **kwargs : Any
            Additional arguments passed to differential_evolution

        Returns
        -------
        OptimizationRun
            Result of the optimization run
        """
        # Solver options from the speed preset, then per-call overrides
        de_kwargs: dict[str, Any] = dict(self._solver_opts)
        de_kwargs.update(kwargs)
        de_kwargs.setdefault("strategy", "best1bin")
        de_kwargs.setdefault("disp", False)

        # workers != 1 (including -1 for all CPUs) requires deferred updating,
        # unless the caller states an updating mode of its own.
        workers = int(de_kwargs.get("workers", 1))
        de_kwargs.setdefault("updating", "deferred" if workers != 1 else "immediate")

        # SciPy DE popsize is a multiplier of the free parameter dimension:
        # the population holds popsize * (N - N_equal) members, so dimensions
        # pinned to lb == ub do not count. Interpret the config popsize as the
        # absolute population size (MATLAB-like) and convert.
        n_params = max(1, sum(1 for lower, upper in self.bounds if upper > lower))
        target_pop = int(de_kwargs.get("popsize", 15))
        pop_multiplier = int(np.ceil(target_pop / n_params))
        de_kwargs["popsize"] = max(5, pop_multiplier)

        if seed is not None:
            de_kwargs["seed"] = seed

        # Create callback wrapper if needed
        callback_wrapper = None
        if self.callback is not None:
            user_callback = self.callback  # local binding for mypy narrowing

            def callback_wrapper(xk: NDArray, convergence: float) -> bool:
                user_callback(xk, convergence)
                return False  # Don't stop early

        # Run differential evolution
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_SCIPY_DE_WARNING, category=UserWarning)
            result: OptimizeResult = differential_evolution(
                self.objective,
                self.bounds,
                callback=callback_wrapper,
                **de_kwargs,
            )

        # RMSE for acceptance should be computed in Volts from the OCV term,
        # not derived from the weighted objective.
        if self.rmse_fn is None:
            rmse = float(np.sqrt(result.fun))
        else:
            rmse = float(self.rmse_fn(result.x))

        return OptimizationRun(
            params=result.x,
            cost=result.fun,
            rmse=rmse,
            success=result.success,
            n_iterations=result.nit,
            n_function_evals=result.nfev,
        )

    def run(
        self,
        req_accepted: int | None = None,
        max_tries: int | None = None,
        rmse_threshold: float | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
        **kwargs: Any,
    ) -> MultiRunResult:
        """Run multi-run optimization.

        Performs multiple optimization runs until req_accepted runs with
        RMSE < threshold are obtained, or max_tries is reached.

        Parameters
        ----------
        req_accepted : int, optional
            Number of required accepted runs, by default from config
        max_tries : int, optional
            Maximum number of optimization attempts, by default from config
        rmse_threshold : float, optional
            RMSE threshold for accepting a run, by default from config
        progress_callback : Callable, optional
            Called after each run with (accepted_count, rejected_count, run_number)
        **kwargs : Any
            Additional arguments passed to differential_evolution

        Returns
        -------
        MultiRunResult
            Result containing best parameters and statistics

        Notes
        -----
        The multi-run strategy helps ensure robust results by:
        1. Running multiple optimizations with different random seeds
        2. Only accepting runs that meet the RMSE threshold
        3. Computing statistics across accepted runs

        This approach helps identify the global optimum in a complex,
        multi-modal parameter space.
        """
        # Use config defaults if not specified
        if req_accepted is None:
            req_accepted = self.config.req_accepted
        if max_tries is None:
            max_tries = self.config.max_tries_overall
        if rmse_threshold is None:
            rmse_threshold = self.config.rmse_threshold

        accepted_runs: list[OptimizationRun] = []
        rejected_runs: list[OptimizationRun] = []

        run_number = 0
        while len(accepted_runs) < req_accepted and run_number < max_tries:
            # Use different seed for each run
            seed = self.config.random_seed
            if seed is not None:
                seed = seed + run_number

            # Run single optimization
            run_result = self._run_single(seed=seed, **kwargs)
            run_number += 1

            # Check acceptance criterion
            if run_result.rmse < rmse_threshold:
                accepted_runs.append(run_result)
            else:
                rejected_runs.append(run_result)

            # Call progress callback (before built-in print so headers appear first)
            if progress_callback is not None:
                progress_callback(len(accepted_runs), len(rejected_runs), run_number)

            if self.config.print_progress:
                status = "accepted" if run_result.rmse < rmse_threshold else "rejected"
                rmse_mv = run_result.rmse * 1000.0
                print(
                    f"Run {run_number}: RMSE={rmse_mv:.1f} mV ({status}) "
                    f"[accepted={len(accepted_runs)}, rejected={len(rejected_runs)}]"
                )

        # Find best result
        if accepted_runs:
            best_run = min(accepted_runs, key=lambda r: r.rmse)

            # Compute statistics across accepted runs
            all_params = np.array([r.params for r in accepted_runs])
            std_params = np.std(all_params, axis=0)
        else:
            # Fall back to best rejected run if no accepted runs
            if rejected_runs:
                best_run = min(rejected_runs, key=lambda r: r.rmse)
                rmse_mv = best_run.rmse * 1000.0
                warnings.warn(
                    f"No runs met RMSE threshold ({rmse_threshold:.6f} V). "
                    f"Best RMSE: {rmse_mv:.1f} mV",
                    UserWarning,
                )
            else:
                raise RuntimeError("No optimization runs completed")
            std_params = None

        return MultiRunResult(
            best_params=best_run.params,
            best_cost=best_run.cost,
            best_rmse=best_run.rmse,
            accepted_runs=accepted_runs,
            rejected_runs=rejected_runs,
            std_params=std_params,
        )
