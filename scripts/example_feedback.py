"""Example script for solving the feedback control problem."""

import copy

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config

from rspmp.dynamics import FeedbackDynamics
from rspmp.problems import FeedbackOptimalControlProblem, problem_parameters
from rspmp.solvers import IndirectFeedbackSolver, solver_parameters

config.update("jax_enable_x64", True)

np.random.seed(0)


dynamics = FeedbackDynamics()
n_x, n_u = dynamics.num_states, dynamics.num_controls
final_time = 2.0
N = 40
M = 10
dt = final_time / N
times = dt * jnp.arange(N + 1)
initial_state = jnp.array([-1, -4.5, 4.5]) * jnp.pi / 180.0
DWs = np.sqrt(dt) * np.random.randn(M, N, n_x)  # Samples from Brownian motion


problem_parameters = copy.deepcopy(problem_parameters)
problem_parameters["discretization_time"] = dt
problem_parameters["sample_size"] = M
problem_parameters["horizon"] = N
problem_parameters["initial_state"] = initial_state
problem_parameters["DWs"] = DWs
ocp = FeedbackOptimalControlProblem(dynamics, problem_parameters)


solver_parameters["num_iterations_max"] = 100
solver_parameters["tolerance"] = 1e-9
solver = IndirectFeedbackSolver(ocp, solver_parameters)


initial_costates_guess = ocp.initial_guess_initial_adjoints()
initial_adjoints = initial_costates_guess
control_penalization_goal = 3.0
control_penalizations = (
    np.linspace(
        np.sqrt(50 * control_penalization_goal), np.sqrt(control_penalization_goal), 200
    )
    ** 2
)
for control_penalization in control_penalizations:
    print("Solving for control_penalization =", control_penalization)
    problem_parameters["control_quad_penalization_scalar"] = control_penalization
    solver_output = solver.solve(
        initial_costates_guess=initial_adjoints,
        problem_params=problem_parameters,
        verbose=True,
    )
    initial_adjoints = solver_output[2]
states, controls, initial_adjoints, convergence_error, solve_time = solver_output
costates = solver.times_states_controls_costates(initial_adjoints, ocp.params)[-1]


times_states = jnp.repeat(times[np.newaxis], repeats=M, axis=0)
times_controls = times[:-1]
colors = ["r", "g", "b"]

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
for dim in range(n_x):
    plt.plot(times_states.T, states[..., dim].T, c=colors[dim], alpha=0.3, linewidth=2)
plt.grid(True)
plt.xlabel("Time [s]")
plt.title("Optimal State Trajectory")
plt.subplot(1, 3, 2)
for dim in range(n_u):
    plt.plot(
        times_controls,
        (controls[:, dim] * states[:, :-1, dim]).T,
        c=colors[dim],
        alpha=0.3,
        linewidth=2,
    )
plt.grid(True)
plt.xlabel("Time [s]")
plt.title("Optimal Control Trajectory")
plt.tight_layout()
plt.subplot(1, 3, 3)
for dim in range(n_x):
    plt.plot(
        times_states.T, costates[..., dim].T, c=colors[dim], alpha=0.3, linewidth=2
    )
plt.grid(True)
plt.xlabel("Time [s]")
plt.title("Optimal Co-State Trajectory")
plt.show()
