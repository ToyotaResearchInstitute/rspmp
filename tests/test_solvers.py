"""Tests solvers."""

import copy

import jax.numpy as jnp
import numpy as np
import pytest

from rspmp.dynamics import Dynamics
from rspmp.problems import (
    FeedbackOptimalControlProblem,
    OptimalControlProblem,
    problem_parameters,
)
from rspmp.solvers import IndirectSolver, SQPSolver, solver_parameters

np.random.seed(0)


@pytest.fixture
def optimal_control_problem():
    dynamics = Dynamics()
    n_x = dynamics.num_states
    final_time, N, M = 3.0, 10, 4
    dt = final_time / N
    # Initial / final states
    initial_state = jnp.array([-1, -4.5, 4.5]) * jnp.pi / 180.0
    # Samples from Brownian motion
    DWs = np.sqrt(dt) * np.random.randn(M, N, n_x)
    new_problem_parameters = copy.deepcopy(problem_parameters)
    new_problem_parameters["discretization_time"] = dt
    new_problem_parameters["sample_size"] = M
    new_problem_parameters["horizon"] = N
    new_problem_parameters["initial_state"] = initial_state
    new_problem_parameters["DWs"] = DWs
    ocp = OptimalControlProblem(dynamics, new_problem_parameters)
    return ocp


@pytest.fixture
def feedback_optimal_control_problem():
    dynamics = Dynamics()
    n_x = dynamics.num_states
    final_time, N, M = 3.0, 40, 3
    dt = final_time / N
    # Initial / final states
    initial_state = jnp.array([-1, -4.5, 4.5]) * jnp.pi / 180.0
    # Samples from Brownian motion
    DWs = np.sqrt(dt) * np.random.randn(M, N, n_x)
    new_problem_parameters = copy.deepcopy(problem_parameters)
    new_problem_parameters["control_quad_penalization_scalar"] = 4.0
    new_problem_parameters["discretization_time"] = dt
    new_problem_parameters["sample_size"] = M
    new_problem_parameters["horizon"] = N
    new_problem_parameters["initial_state"] = initial_state
    new_problem_parameters["DWs"] = DWs
    ocp = FeedbackOptimalControlProblem(dynamics, new_problem_parameters)
    return ocp


def verify_solver_on_problem(solver, problem: OptimalControlProblem):
    """General checks for solvers on particular problems."""
    solver_output = solver.solve(verbose=True)
    states, controls, initial_adjoints, convergence_error, solve_time = solver_output

    dynamics = problem.dynamics
    assert convergence_error < 1e-6
    assert isinstance(solve_time, float)
    assert states.shape[0] == problem.sample_size
    assert states.shape[1] == problem.horizon + 1
    assert states.shape[2] == dynamics.num_states
    assert controls.shape[0] == problem.horizon
    assert controls.shape[1] == dynamics.num_controls
    assert initial_adjoints.shape[0] == problem.sample_size
    assert initial_adjoints.shape[1] == dynamics.num_states


def test_solve_optimal_control_problem_with_sqp(optimal_control_problem):
    """Tests sqp solver on optimal control problem."""
    solver = SQPSolver(optimal_control_problem, solver_parameters)
    verify_solver_on_problem(solver, optimal_control_problem)


def test_solve_optimal_control_problem_with_indirect(optimal_control_problem):
    """Tests sqp solver on optimal control problem."""
    solver = IndirectSolver(optimal_control_problem, solver_parameters)
    verify_solver_on_problem(solver, optimal_control_problem)
