"""Comparisons of solutions from direct and indirect methods for different parameters."""

import argparse
import copy
import pickle

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jax import config
from matplotlib import rc, rcParams
from tqdm import tqdm

from rspmp.dynamics import Dynamics
from rspmp.problems import OptimalControlProblem, problem_parameters
from rspmp.solvers import IndirectSolver, SQPSolver, solver_parameters
from rspmp.validation import MonteCarloOptimalControlValidation, validation_parameters

config.update("jax_enable_x64", True)

np.random.seed(0)

dynamics = Dynamics()
n_x, n_u = dynamics.num_states, dynamics.num_controls
final_time = 3.0
problem_parameters["initial_state"] = jnp.array([-1, -4.5, 4.5]) * jnp.pi / 180.0


def solve_problems_comparison():
    """Comparison between the direct and indirect methods for different parameters."""
    num_repeats = 20
    sample_sizes = [5, 10, 20, 30, 40, 50]
    horizons = [10, 20, 30, 40]
    computation_times = {
        "sqp": np.zeros((num_repeats, len(sample_sizes), len(horizons))),
        "pmp": np.zeros((num_repeats, len(sample_sizes), len(horizons))),
    }

    for sample_size_i, sample_size in enumerate(pbar := tqdm(sample_sizes)):
        pbar.set_description("Solving for sample size " + str(sample_size))

        for horizon_i, horizon in enumerate(horizons):
            # setup problem
            dt = final_time / horizon
            DWs = np.sqrt(dt) * np.random.randn(sample_size, horizon, n_x)
            problem_parameters["discretization_time"] = dt
            problem_parameters["sample_size"] = sample_size
            problem_parameters["horizon"] = horizon
            problem_parameters["DWs"] = DWs
            ocp = OptimalControlProblem(dynamics, problem_parameters, verbose=False)
            # setup solvers
            solver_sqp = SQPSolver(ocp, solver_parameters, verbose=False)
            solver_indirect = IndirectSolver(ocp, solver_parameters, verbose=False)
            for repeat in range(num_repeats):
                # setup problem
                dt = final_time / horizon
                DWs = np.sqrt(dt) * np.random.randn(sample_size, horizon, n_x)
                problem_parameters["discretization_time"] = dt
                problem_parameters["sample_size"] = sample_size
                problem_parameters["horizon"] = horizon
                problem_parameters["DWs"] = DWs
                # solve problem
                solver_output = solver_sqp.solve(problem_params=problem_parameters)
                _, _, _, _, solve_time_sqp = solver_output
                solver_output = solver_indirect.solve(problem_params=problem_parameters)
                _, _, _, _, solve_time_pmp = solver_output
                computation_times["sqp"][
                    repeat, sample_size_i, horizon_i
                ] = solve_time_sqp
                computation_times["pmp"][
                    repeat, sample_size_i, horizon_i
                ] = solve_time_pmp

    results = {
        "num_repeats": num_repeats,
        "sample_sizes": sample_sizes,
        "horizons": horizons,
        "computation_times": computation_times,
    }
    with open("results/comparison.pkl", "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def solve_problems_large_sample_sizes_indirect():
    """Evaluates sensitivity to the sample size."""
    num_repeats = 20
    sample_sizes = [1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100]
    horizon = 40
    controls = {"pmp": np.zeros((num_repeats, len(sample_sizes), horizon, n_u))}
    computation_times = {"pmp": np.zeros((num_repeats, len(sample_sizes)))}
    costs = {"pmp": np.zeros((num_repeats, len(sample_sizes)))}

    accurate_solver_parameters = copy.deepcopy(solver_parameters)
    accurate_solver_parameters["tolerance"] = 1e-9
    for sample_size_i, sample_size in enumerate(pbar := tqdm(sample_sizes)):
        pbar.set_description("Solving for sample size " + str(sample_size))
        # setup problem
        dt = final_time / horizon
        DWs = np.sqrt(dt) * np.random.randn(sample_size, horizon, n_x)
        problem_parameters["discretization_time"] = dt
        problem_parameters["sample_size"] = sample_size
        problem_parameters["horizon"] = horizon
        problem_parameters["DWs"] = DWs
        ocp = OptimalControlProblem(dynamics, problem_parameters, verbose=False)
        # setup solvers
        solver = IndirectSolver(ocp, accurate_solver_parameters, verbose=False)
        for repeat in range(num_repeats):
            # setup problem
            DWs = np.sqrt(dt) * np.random.randn(sample_size, horizon, n_x)
            problem_parameters["DWs"] = DWs
            # solve problem
            solver_output = solver.solve(problem_params=problem_parameters)
            _, controls_solution, _, _, solve_time = solver_output
            computation_times["pmp"][repeat, sample_size_i] = solve_time
            controls["pmp"][repeat, sample_size_i] = np.array(controls_solution)

    # validate
    validator = MonteCarloOptimalControlValidation(
        dynamics=dynamics,
        problem_type=OptimalControlProblem,
        problem_parameters=problem_parameters,
        parameters=validation_parameters,
    )
    for sample_size_i, sample_size in enumerate(sample_sizes):
        for repeat in range(num_repeats):
            costs["pmp"][repeat, sample_size_i] = validator.get_solution_cost(
                controls["pmp"][repeat, sample_size_i]
            )
    computation_times["pmp"] = np.mean(computation_times["pmp"], axis=0)
    results = {
        "num_repeats": num_repeats,
        "sample_sizes": sample_sizes,
        "horizon": horizon,
        "computation_times": computation_times,
        "costs": costs,
    }
    with open("results/sample_sizes_sweep.pkl", "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot():
    """Plots the results."""
    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 14
    ticks_fontsize = 25
    rc("text", usetex=True)
    with open("results/comparison.pkl", "rb") as handle:
        results = pickle.load(handle)
        sample_sizes = results["sample_sizes"]
        horizons = results["horizons"]
        computation_times = results["computation_times"]
        computation_times = {
            "sqp": np.median(computation_times["sqp"], axis=0),
            "pmp": np.median(computation_times["pmp"], axis=0),
        }
    plt.figure(figsize=(12, 5))
    cmap = plt.get_cmap("jet")  # winter, plasma, rainbow
    min_h, max_h = np.min(horizons), np.max(horizons)
    colors = [cmap((val - min_h) / (max_h - min_h)) for val in horizons]
    solvers = ["sqp", "pmp"]
    markers = ["x", "o"]
    linestyles = ["--", "-"]
    for solver, ls, marker in zip(solvers, linestyles, markers):
        for horizon_i in range(len(horizons)):
            plt.plot(
                sample_sizes,
                computation_times[solver][..., horizon_i],
                c=colors[horizon_i],
                marker=marker,
                markersize=10,
                linewidth=2,
                linestyle=ls,
            )
            plt.scatter(
                sample_sizes,
                computation_times[solver][..., horizon_i],
                c=colors[horizon_i],
                marker=marker,
                s=130,
                linewidth=2,
            )
    plt.yscale("log")
    xlims = plt.xlim()
    ylims = plt.ylim()
    # colorbar for horizons
    ctf = plt.contourf(
        [[0, 0], [0, 0]],
        np.linspace(np.min(horizons), np.max(horizons), num=201),
        cmap=cmap,
    )
    cbar = plt.colorbar(ctf, ticks=horizons)
    cbar.set_label(r"$N$", fontsize=ticks_fontsize + 4, rotation="horizontal")
    cbar.ax.tick_params(labelsize=ticks_fontsize)
    # legend
    lines = [
        matplotlib.lines.Line2D(
            [0], [0], color="k", linewidth=3, linestyle=ls, marker=marker, markersize=14
        )
        for (ls, marker) in zip(linestyles, markers)
    ]
    plt.legend(lines, [r"Direct", r"Indirect"], fontsize=ticks_fontsize)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xticks(sample_sizes, sample_sizes, fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.title(r"Computation Time ($s$)", fontsize=28)
    plt.xlabel(r"Sample Size $M$", fontsize=28)
    plt.grid(True)
    plt.grid(which="minor", alpha=0.75, linestyle="--")
    plt.grid(which="major", alpha=1.0, linestyle=":")
    plt.tight_layout()
    plt.savefig("results/comparison.png")

    speedups = np.zeros((len(sample_sizes), len(horizons)))
    for horizon_i in range(len(horizons)):
        for sample_size_i in range(len(sample_sizes)):
            speedups[sample_size_i, horizon_i] = (
                computation_times["sqp"][sample_size_i, horizon_i]
                / computation_times["pmp"][sample_size_i, horizon_i]
            )
    print("average speedup =", np.mean(speedups))

    with open("results/sample_sizes_sweep.pkl", "rb") as handle:
        results = pickle.load(handle)
        # num_repeats = results["num_repeats"]
        sample_sizes = results["sample_sizes"]
        computation_times = results["computation_times"]
        costs = results["costs"]
        costs_means = {"pmp": np.median(costs["pmp"], axis=0)}
        costs_stds = {
            "pmp": np.median(np.abs(costs["pmp"] - costs_means["pmp"]), axis=0)
        }
    plt.figure(figsize=(8, 5))
    for solver, ls, marker in zip(["pmp"], ["-"], ["o"]):
        plt.plot(
            sample_sizes,
            costs_means[solver],
            color="b",
            marker=marker,
            linewidth=2,
            linestyle=ls,
        )
        plt.fill_between(
            sample_sizes,
            costs_means[solver] - costs_stds[solver],
            costs_means[solver] + costs_stds[solver],
            color="b",
            alpha=0.1,
        )

    # sqrt x scale
    def func(e):
        return e**0.5

    def inv_func(e):
        return e**2

    plt.xscale(matplotlib.scale.FuncScale(plt.gca(), (func, inv_func)))
    plt.xticks(sample_sizes, sample_sizes, fontsize=ticks_fontsize)
    # plt.yticks(fontsize=ticks_fontsize)
    plt.yticks(
        [0.14, 0.15, 0.16, 0.17], [0.14, 0.15, 0.16, 0.17], fontsize=ticks_fontsize
    )
    plt.title(r"Cost", fontsize=28)
    plt.xlabel(r"Sample Size $M$", fontsize=28)
    plt.grid(True)
    plt.grid(which="minor", alpha=0.75, linestyle="--")
    plt.grid(which="major", alpha=1.0, linestyle=":")
    plt.tight_layout()
    plt.savefig("results/sample_sizes_sweep.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solve-comparison",
        action="store_true",
        default=False,
        help="Compute solutions with different methods.",
    )
    parser.add_argument(
        "--sample-sizes-sweep",
        action="store_true",
        default=False,
        help="Solve the problem for different samples sizes.",
    )
    parser.add_argument(
        "--plot", action="store_true", default=True, help="Plot solutions."
    )
    args = parser.parse_args()

    if args.solve_comparison:
        solve_problems_comparison()
    if args.sample_sizes_sweep:
        solve_problems_large_sample_sizes_indirect()
    if args.plot:
        plot()
