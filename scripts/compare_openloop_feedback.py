"""Scripts to compare the solutions of the open loop and feedback problems."""

import argparse
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config
from matplotlib import rc, rcParams
from tqdm import tqdm

from rspmp.dynamics import Dynamics, FeedbackDynamics
from rspmp.problems import (
    FeedbackOptimalControlProblem,
    OptimalControlProblem,
    problem_parameters,
)
from rspmp.solvers import IndirectFeedbackSolver, IndirectSolver, solver_parameters

# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

np.random.seed(0)

control_penalization_weight = 3.0  # R

dynamics = FeedbackDynamics()
n_x, n_u = dynamics.num_states, dynamics.num_controls
final_time = 2.0
N = 40
M = 10
dt = final_time / N
times = dt * jnp.arange(N + 1)
initial_state = jnp.array([-1, -4.5, 4.5]) * jnp.pi / 180.0  # Initial states
DWs = np.sqrt(dt) * np.random.randn(M, N, n_x)  # Samples from Brownian motion


def solve_problems():
    problem_parameters["discretization_time"] = dt
    problem_parameters["sample_size"] = M
    problem_parameters["horizon"] = N
    problem_parameters["initial_state"] = initial_state
    problem_parameters["DWs"] = DWs
    problem_parameters["control_quad_penalization_scalar"] = control_penalization_weight

    solver_parameters["num_iterations_max"] = 100
    solver_parameters["tolerance"] = 1e-9

    results = {}

    print("--------------------------")
    print("Solving open-loop problem.")
    print(
        "Directly solving for R =",
        problem_parameters["control_quad_penalization_scalar"],
    )
    ocp_openloop = OptimalControlProblem(Dynamics(), problem_parameters)
    solver_openloop = IndirectSolver(ocp_openloop, solver_parameters)
    solver_output_openloop = solver_openloop.solve()
    states, controls, initial_adjoints, _, _ = solver_output_openloop
    costates = solver_openloop.times_states_controls_costates(
        initial_adjoints, problem_parameters
    )[-1]
    results["openloop"] = {
        "states": states,  # (M, N+1, n_x)
        "controls": controls,  # (N, n_u)
        "costates": costates,  # (M, N+1, n_x)
    }
    print("--------------------------")

    print("--------------------------")
    print("Solving feedback problem. ")
    ocp = FeedbackOptimalControlProblem(dynamics, problem_parameters)
    solver = IndirectFeedbackSolver(ocp, solver_parameters)
    initial_costates_guess = ocp.initial_guess_initial_adjoints()
    initial_adjoints = initial_costates_guess
    # solve via a homotopy method on R
    start_control_penalization_weight = 100.0  # 50. * control_penalization_weight
    control_penalization_weights = np.arange(
        start_control_penalization_weight, control_penalization_weight, -0.1
    )
    control_penalization_weights[-1] = control_penalization_weight
    for _, weight in enumerate(pbar := tqdm(control_penalization_weights)):
        pbar.set_description("Homotopy: Solving for R = " + str(np.round(weight, 1)))
        problem_parameters["control_quad_penalization_scalar"] = weight
        solver_output = solver.solve(
            initial_costates_guess=initial_adjoints,
            problem_params=problem_parameters,
            verbose=False,
        )
        initial_adjoints = solver_output[2]
    states, controls, initial_adjoints, convergence_error, solve_time = solver_output
    costates = solver.times_states_controls_costates(initial_adjoints, ocp.params)[-1]
    results["feedback"] = {
        "states": states,  # (M, N+1, n_x)
        "controls": controls,  # (N, n_u)
        "costates": costates,  # (M, N+1, n_x)
    }
    print("--------------------------------------------------")

    with open("results/openloop_feedback.pkl", "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot():
    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 14
    ticks_fontsize = 25
    time_ticks = [0, 0.5, 1, 1.5, 2]
    rc("text", usetex=True)

    with open("results/openloop_feedback.pkl", "rb") as handle:
        results = pickle.load(handle)

    times_states = jnp.repeat(times[np.newaxis], repeats=M, axis=0)
    times_controls = times[:-1]
    colors = ["r", "g", "b"]
    solvers = ["openloop", "feedback"]

    # feedback u = K x
    results["feedback"]["controls"] = (
        results["feedback"]["controls"] * results["feedback"]["states"][:, :-1]
    )

    xlims = {"states": [None, None], "controls": [None, None], "costates": [None, None]}
    ylims = {"states": [None, None], "controls": [None, None], "costates": [None, None]}
    plt.figure(figsize=(16, 5))
    for solver_i, solver in enumerate(solvers):
        states = results[solver]["states"]
        controls = results[solver]["controls"]
        costates = results[solver]["costates"]

        plt.subplot(2, 3, 3 * solver_i + 1)
        for dim in range(n_x):
            plt.plot(
                times_states.T,
                states[..., dim].T,
                c=colors[dim],
                alpha=0.3,
                linewidth=2,
            )
        if solver_i == 0:
            plt.title(r"State Trajectories $x_t$", fontsize=28)

        plt.subplot(2, 3, 3 * solver_i + 2)
        for dim in range(n_u):
            if solver_i == 0:
                alpha = 1
            elif solver_i == 1:
                alpha = 0.3

            plt.plot(
                times_controls,
                controls[..., dim].T,
                c=colors[dim],
                alpha=alpha,
                linewidth=2,
            )
        if solver_i == 0:
            plt.title(r"Control Trajectories $u_t$", fontsize=28)
        elif solver_i == 1:
            plt.xlabel(r"$t$", fontsize=28)

        plt.subplot(2, 3, 3 * solver_i + 3)
        for dim in range(n_x):
            plt.plot(
                times_states.T,
                costates[..., dim].T,
                c=colors[dim],
                alpha=0.3,
                linewidth=2,
            )
        if solver_i == 0:
            plt.title(r"Adjoint Trajectories $p_t$", fontsize=28)

        for plot_i, variable in enumerate(["states", "controls", "costates"]):
            plt.subplot(2, 3, 3 * solver_i + 1 + plot_i)
            plt.grid(True)
            if solver_i == 1:
                plt.xlabel(r"$t$", fontsize=28)
            if solver_i == 0:
                plt.xticks(time_ticks, [], fontsize=ticks_fontsize)
            plt.xticks(fontsize=ticks_fontsize)
            plt.yticks(fontsize=ticks_fontsize)
            if solver_i == 0:
                xlims[variable] = list(plt.xlim())
                ylims[variable] = list(plt.ylim())
            elif solver_i == 1:
                xlims[variable][0] = min(plt.xlim()[0], xlims[variable][0])
                xlims[variable][1] = max(plt.xlim()[1], xlims[variable][1])
                ylims[variable][0] = min(plt.ylim()[0], ylims[variable][0])
                ylims[variable][1] = max(plt.ylim()[1], ylims[variable][1])
    for solver_i, _ in enumerate(solvers):
        for plot_i, variable in enumerate(["states", "controls", "costates"]):
            plt.subplot(2, 3, 3 * solver_i + 1 + plot_i)
            plt.xlim(xlims[variable])
            plt.ylim(ylims[variable])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.205, hspace=0.126)
    plt.savefig("results/openloop_feedback.png")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solve",
        action="store_true",
        default=False,
        help="Solve openloop and feedback problems.",
    )
    parser.add_argument(
        "--plot", action="store_true", default=True, help="Plot solutions."
    )
    args = parser.parse_args()

    if args.solve:
        solve_problems()
    if args.plot:
        plot()
