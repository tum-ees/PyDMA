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
from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, OptimizeResult

from pydma.utils.dma_config import DMAConfig


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
    """Best parameters from accepted runs."""

    best_cost: float
    """Best cost value."""

    best_rmse: float
    """Best RMSE value."""

    accepted_runs: list[OptimizationRun] = field(default_factory=list)
    """List of accepted runs (RMSE < threshold)."""

    rejected_runs: list[OptimizationRun] = field(default_factory=list)
    """List of rejected runs (RMSE >= threshold)."""

    mean_params: NDArray[np.floating] | None = None
    """Mean parameters from accepted runs."""

    std_params: NDArray[np.floating] | None = None
    """Standard deviation of parameters from accepted runs."""

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
        # Merge solver options with any overrides
        de_kwargs = {
            "strategy": self._solver_opts.get("strategy", "best1bin"),
            "maxiter": self._solver_opts.get("maxiter", 1000),
            "popsize": self._solver_opts.get("popsize", 500),
            "tol": self._solver_opts.get("tol", 0.01),
            "mutation": self._solver_opts.get("mutation", (0.5, 1.0)),
            "recombination": self._solver_opts.get("recombination", 0.7),
            "workers": self._solver_opts.get("workers", 1),
            "updating": "deferred" if self._solver_opts.get("workers", 1) > 1 else "immediate",
            "disp": False,
        }
        de_kwargs.update(kwargs)

        # SciPy DE popsize is a multiplier of parameter dimension.
        # Interpret config popsize as absolute population size (MATLAB-like),
        # then convert to DE multiplier.
        n_params = max(1, len(self.bounds))
        target_pop = int(de_kwargs.get("popsize", 15))
        pop_multiplier = int(np.ceil(target_pop / n_params))
        de_kwargs["popsize"] = max(5, pop_multiplier)

        if seed is not None:
            de_kwargs["seed"] = seed

        # Create callback wrapper if needed
        callback_wrapper = None
        if self.callback is not None:
            def callback_wrapper(xk: NDArray, convergence: float) -> bool:
                self.callback(xk, convergence)
                return False  # Don't stop early

        # Run differential evolution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

            if self.config.print_progress:
                status = "accepted" if run_result.rmse < rmse_threshold else "rejected"
                rmse_mv = run_result.rmse * 1000.0
                print(
                    f"Run {run_number}: RMSE={rmse_mv:.1f} mV ({status}) "
                    f"[accepted={len(accepted_runs)}, rejected={len(rejected_runs)}]"
                )

            # Call progress callback
            if progress_callback is not None:
                progress_callback(len(accepted_runs), len(rejected_runs), run_number)

        # Find best result
        if accepted_runs:
            best_run = min(accepted_runs, key=lambda r: r.cost)

            # Compute statistics across accepted runs
            all_params = np.array([r.params for r in accepted_runs])
            mean_params = np.mean(all_params, axis=0)
            std_params = np.std(all_params, axis=0)
        else:
            # Fall back to best rejected run if no accepted runs
            if rejected_runs:
                best_run = min(rejected_runs, key=lambda r: r.cost)
                rmse_mv = best_run.rmse * 1000.0
                warnings.warn(
                    f"No runs met RMSE threshold ({rmse_threshold:.6f} V). "
                    f"Best RMSE: {rmse_mv:.1f} mV",
                    UserWarning,
                )
            else:
                raise RuntimeError("No optimization runs completed")
            mean_params = None
            std_params = None

        return MultiRunResult(
            best_params=best_run.params,
            best_cost=best_run.cost,
            best_rmse=best_run.rmse,
            accepted_runs=accepted_runs,
            rejected_runs=rejected_runs,
            mean_params=mean_params,
            std_params=std_params,
        )

    def run_single_fast(self, **kwargs: Any) -> OptimizationRun:
        """Run a single fast optimization (useful for debugging).

        Uses reduced population and iterations for quick testing.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments passed to differential_evolution

        Returns
        -------
        OptimizationRun
            Result of the optimization run
        """
        return self._run_single(
            popsize=50,
            maxiter=100,
            **kwargs,
        )


def create_optimizer_from_config(
    config: DMAConfig,
    objective: Callable[[NDArray[np.floating]], float],
    callback: Callable[[NDArray[np.floating], float], None] | None = None,
) -> DMAOptimizer:
    """Create an optimizer from a configuration object.

    Parameters
    ----------
    config : DMAConfig
        Configuration object
    objective : Callable
        Objective function to minimize
    callback : Callable, optional
        Progress callback

    Returns
    -------
    DMAOptimizer
        Configured optimizer instance
    """
    bounds = config.get_bounds()
    return DMAOptimizer(config, objective, bounds, callback)
