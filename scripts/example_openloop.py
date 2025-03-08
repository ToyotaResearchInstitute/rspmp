"""Example script for solving the open loop control problem."""

import copy
from typing import Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config

from rspmp.dynamics import Dynamics
from rspmp.problems import OptimalControlProblem, problem_parameters
from rspmp.solvers import IndirectSolver, SQPSolver, solver_parameters
from rspmp.validation import MonteCarloOptimalControlValidation, validation_parameters

# config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
np.random.seed(0)

dynamics = Dynamics()
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
problem_parameters["control_quad_penalization_scalar"] = 3.0
ocp = OptimalControlProblem(dynamics, problem_parameters)

solver_parameters["num_iterations_max"] = 20
solver_parameters["tolerance"] = 1e-9
solver: Union[SQPSolver, IndirectSolver] = SQPSolver(ocp, solver_parameters)
solver_output = solver.solve(verbose=True)
(
    states_sqp,
    controls_sqp,
    initial_adjoints_sqp,
    convergence_error_sqp,
    solve_time_sqp,
) = solver_output

solver = IndirectSolver(ocp, solver_parameters)
solver_output = solver.solve(verbose=True)
(
    states_pmp,
    controls_pmp,
    initial_adjoints_pmp,
    convergence_error_pmp,
    solve_time_pmp,
) = solver_output
costates_pmp = solver.times_states_controls_costates(initial_adjoints_pmp, ocp.params)[
    -1
]

print("--------------------------------------------------")
print("Solve times:")
print("> SQP:", solve_time_sqp)
print("> PMP:", solve_time_pmp)
print("Difference states:  ", np.linalg.norm(states_sqp - states_pmp))
print("Difference controls:", np.linalg.norm(controls_sqp - controls_pmp))
print("--------------------------------------------------")

validator = MonteCarloOptimalControlValidation(
    dynamics=dynamics,
    problem_type=OptimalControlProblem,
    problem_parameters=problem_parameters,
    parameters=validation_parameters,
)
print("Costs:")
print("> SQP:", validator.get_solution_cost(controls_sqp))
print("> PMP:", validator.get_solution_cost(controls_pmp))
print("--------------------------------------------------")


times_states = jnp.repeat(times[np.newaxis], repeats=M, axis=0)
times_controls = times[:-1]
colors = ["r", "g", "b"]

plt.figure(figsize=(12, 4))
for solver_i in range(2):
    if solver_i == 0:
        states, controls = states_sqp, controls_sqp
    if solver_i == 1:
        states, controls = states_pmp, controls_pmp
    plt.subplot(2, 3, solver_i * 3 + 1)
    for dim in range(n_x):
        plt.plot(
            times_states.T, states[..., dim].T, c=colors[dim], alpha=0.3, linewidth=2
        )
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.title("Optimal State Trajectory")
    plt.subplot(2, 3, solver_i * 3 + 2)
    for dim in range(n_u):
        plt.plot(times_controls, controls[:, dim], c=colors[dim], linewidth=2)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.title("Optimal Control Trajectory")
    plt.tight_layout()
    if solver_i == 1:
        plt.subplot(2, 3, solver_i * 3 + 3)
        for dim in range(n_x):
            plt.plot(
                times_states.T,
                costates_pmp[..., dim].T,
                c=colors[dim],
                alpha=0.3,
                linewidth=2,
            )
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.title("Optimal Co-State Trajectory")
plt.show()
