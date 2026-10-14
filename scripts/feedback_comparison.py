"""Compare the two feedback indirect shooting methods.

Reproduces results in 
T. Lew, "Unified Optimality Conditions for Stochastic Optimal Control
in the Rough Path and Itô Frameworks", 2026,
available at https://arxiv.org/abs/2609.38395.
"""

import argparse
import csv
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config

from rspmp.dynamics import Dynamics, FeedbackDynamics
from rspmp.problems import (
    FeedbackOptimalControlProblem,
    OptimalControlProblem,
    problem_parameters,
)
from rspmp.solvers import (
    IndirectFeedbackConditionalProjectionSolver,
    IndirectFeedbackSolver,
    solver_parameters,
)
from rspmp.validation import MonteCarloOptimalControlValidation

config.update("jax_enable_x64", True)


FINAL_TIME = 2.0
HORIZON = 40
SAMPLE_SIZE = 10
TOLERANCE = 1e-12
MAX_NEWTON_ITERATIONS = 100
CONTROL_WEIGHT = 3.0
CONTROL_WEIGHTS = (
    0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
    3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0,
)
SAMPLE_SIZES = (2, 5, 10, 20, 30, 40, 50)
NUM_TRIALS = 100
VALIDATION_SIZE = 100000
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 16
FIGURE_HEIGHT = 2.75
SHOW_TITLES = False

METHODS = (
    ("old", "Open-Loop Gain Indirect", "#D55E00"),
    ("new", "Feedback Indirect", "#0072B2"),
)
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "feedback_comparison"


def make_parameters(seed, control_weight, sample_size=SAMPLE_SIZE, horizon=HORIZON):
    """Return problem parameters."""
    dynamics = Dynamics()
    dt = FINAL_TIME / horizon
    rng = np.random.default_rng(seed)
    params = problem_parameters.copy()
    params.update(
        discretization_time=dt,
        horizon=horizon,
        sample_size=sample_size,
        initial_state=jnp.array([-1.0, -4.5, 4.5]) * jnp.pi / 180.0,
        DWs=np.sqrt(dt)
        * rng.standard_normal((sample_size, horizon, dynamics.num_states)),
        control_quad_penalization_scalar=control_weight,
    )
    return params


def make_solvers(sample_size=SAMPLE_SIZE):
    """Return solvers."""
    params = make_parameters(0, CONTROL_WEIGHT, sample_size)
    settings = solver_parameters.copy()
    settings.update(
        num_iterations_max=MAX_NEWTON_ITERATIONS,
        tolerance=TOLERANCE,
    )
    return (
        IndirectFeedbackSolver(
            FeedbackOptimalControlProblem(FeedbackDynamics(), params), settings
        ),
        IndirectFeedbackConditionalProjectionSolver(
            OptimalControlProblem(Dynamics(), params), settings
        ),
    )


def solve_newton(solver, params, initial_adjoints):
    """Run Newton method and report whether it converged."""
    adjoints = initial_adjoints
    error = np.inf
    try:
        for _ in range(solver.params["num_iterations_max"]):
            adjoints, residual = solver.newton_step(adjoints, params)
            error = np.linalg.norm(residual, ord=np.inf)
            finite = np.isfinite(error) and np.all(np.isfinite(np.asarray(adjoints)))
            if not finite:
                break
            if error < solver.params["tolerance"]:
                error = np.linalg.norm(
                    solver.shooting_residual(adjoints.ravel(), params), ord=np.inf
                )
                break
    except Exception:
        return False, adjoints
    return np.isfinite(error) and error < solver.params["tolerance"], adjoints


def write_csv(filename, columns, rows):
    with (OUTPUT_DIR / filename).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def run_experiments():
    """Run the convergence and sample-size experiments."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    initial = jnp.zeros((SAMPLE_SIZE, Dynamics().num_states))
    solvers = make_solvers()

    print("Running direct convergence sweep...", flush=True)
    rows = []
    for weight in CONTROL_WEIGHTS:
        print(f"  R={weight:g}: {NUM_TRIALS} trials", flush=True)
        successes = {name: 0 for name, _, _ in METHODS}
        for seed in range(NUM_TRIALS):
            params = make_parameters(seed, weight)
            for (name, _, _), solver in zip(METHODS, solvers):
                converged, _ = solve_newton(solver, params, initial)
                successes[name] += converged
                rows.append((weight, seed, name, int(converged)))
            if (seed + 1) % 10 == 0 and seed + 1 < NUM_TRIALS:
                print(f"    trials {seed + 1}/{NUM_TRIALS}", flush=True)
        print(
            f"    old={successes['old']}/{NUM_TRIALS}, "
            f"new={successes['new']}/{NUM_TRIALS}",
            flush=True,
        )
    write_csv(
        "feedback_convergence_r.csv",
        ("control_weight", "seed", "method", "converged"),
        rows,
    )

    print("Running sample-size sweep for Feedback Indirect...", flush=True)
    validator = MonteCarloOptimalControlValidation(
        OptimalControlProblem,
        Dynamics(),
        make_parameters(0, CONTROL_WEIGHT),
        {"sample_size": VALIDATION_SIZE, "seed": 0},
    )
    test_dws = validator.problem_params["DWs"]
    rows = []
    for sample_size in SAMPLE_SIZES:
        solver = make_solvers(sample_size)[1]
        initial = jnp.zeros((sample_size, Dynamics().num_states))
        print(f"  M={sample_size}: {NUM_TRIALS} seeds", flush=True)
        for seed in range(NUM_TRIALS):
            params = make_parameters(seed, CONTROL_WEIGHT, sample_size)
            converged, adjoints = solve_newton(solver, params, initial)
            cost = np.nan
            try:
                if converged:
                    gains = solver.feedback_gains(adjoints, params)
                    cost = np.mean(validator.get_diagonal_gain_costs(gains, test_dws))
            except Exception as exception:
                print(f"    seed={seed}: cost evaluation failed: {exception}")
            rows.append((sample_size, seed, int(converged), cost))
            status = "OK" if converged else "FAIL"
            print(
                f"    seed {seed + 1}/{NUM_TRIALS}: {status}",
                flush=True,
            )
    write_csv(
        "feedback_sample_size.csv",
        ("sample_size", "seed", "converged", "cost"),
        rows,
    )


def convergence_rates(rows, method):
    """Return the empirical convergence rate at each control weight."""
    rows = [row for row in rows if row["method"] == method]
    xs = sorted({float(row["control_weight"]) for row in rows})
    rates = []
    for x in xs:
        flags = np.array(
            [int(row["converged"]) for row in rows if float(row["control_weight"]) == x]
        )
        rates.append(flags.mean())
    return np.array(xs), np.array(rates)


def format_axis(axis, xlabel, ylabel, title):
    axis.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    if SHOW_TITLES:
        axis.set_title(title, fontsize=TITLE_FONTSIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    axis.minorticks_off()
    axis.grid(which="major", axis="both", alpha=0.3)


def plot_results():
    """Generate figures from the saved CSV files."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "text.usetex": True})

    with (OUTPUT_DIR / "feedback_convergence_r.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    figure, axis = plt.subplots(figsize=(7.0, FIGURE_HEIGHT))
    for name, label, color in METHODS:
        x, rate = convergence_rates(rows, name)
        axis.plot(
            x,
            rate,
            "-o",
            markersize=6,
            linewidth=2,
            color=color,
            label=label,
        )
    axis.set_ylim(-0.05, 1.05)
    format_axis(axis, "Control Weight $R$", "Convergence Rate", "Direct Solve")
    axis.set_xticks(x[::2])
    axis.set_xticks(x, minor=True)
    axis.tick_params(axis="x", which="minor", length=3)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles[::-1],
        labels[::-1],
        frameon=False,
        loc="lower right",
        fontsize=LEGEND_FONTSIZE,
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"feedback_convergence_r.png", bbox_inches="tight", dpi=300)
    plt.close(figure)

    with (OUTPUT_DIR / "feedback_sample_size.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    sample_sizes = sorted({int(row["sample_size"]) for row in rows})
    means, errors = [], []
    for sample_size in sample_sizes:
        costs = np.array(
            [
                float(row["cost"])
                for row in rows
                if int(row["sample_size"]) == sample_size
                and np.isfinite(float(row["cost"]))
            ]
        )
        means.append(costs.mean() if len(costs) else np.nan)
        errors.append(
            1.96 * costs.std(ddof=1) / np.sqrt(len(costs)) if len(costs) > 1 else 0.0
        )
    means, errors = np.array(means), np.array(errors)
    figure, axis = plt.subplots(figsize=(7.0, FIGURE_HEIGHT))
    axis.fill_between(
        sample_sizes, means - errors, means + errors, color="b", alpha=0.1
    )
    axis.errorbar(
        sample_sizes,
        means,
        yerr=errors,
        fmt="-o",
        color="b",
        linewidth=2,
        capsize=4,
    )
    axis.set_xticks(sample_sizes)
    format_axis(axis, "Sample Size $M$", "Cost", "Feedback Indirect")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"feedback_sample_size.png", bbox_inches="tight", dpi=300)
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        run_experiments()
    plot_results()
    print(f"Saved results in {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
