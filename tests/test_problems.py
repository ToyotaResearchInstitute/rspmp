"""Tests problems."""

import jax.numpy as jnp

from rspmp.dynamics import Dynamics, FeedbackDynamics
from rspmp.problems import (
    FeedbackOptimalControlProblem,
    OptimalControlProblem,
    problem_parameters,
)


def verify_problem(problem: OptimalControlProblem):
    """General checks for problems."""
    assert isinstance(problem, OptimalControlProblem)
    dynamics = problem.dynamics

    dt = problem.dt
    horizon = problem.horizon
    sample_size = problem.sample_size
    num_variables = problem.num_variables

    assert isinstance(dt, float)
    assert isinstance(horizon, int)
    assert isinstance(sample_size, int)
    assert isinstance(num_variables, int)
    assert dt >= 1e-6
    assert horizon >= 1
    assert sample_size >= 1
    assert num_variables >= 1

    (M, N, num_states, num_controls, nv) = problem.dimensions

    assert M == sample_size
    assert N == horizon
    assert nv == num_variables
    assert num_states == dynamics.num_states
    assert num_controls == dynamics.num_controls

    opt_vars = problem.initial_guess_optvars()
    assert len(opt_vars) == num_variables

    states, controls = problem.convert_optvars_to_states_controls(opt_vars)
    assert states.shape[0] == sample_size
    assert states.shape[1] == horizon + 1
    assert states.shape[2] == num_states
    assert controls.shape[0] == horizon
    assert controls.shape[1] == num_controls

    initial_constraints = problem.initial_constraints(opt_vars)
    assert len(initial_constraints) == sample_size * num_states

    dynamics_constraints = problem.dynamics_constraints(opt_vars, problem.params)
    assert len(dynamics_constraints) == sample_size * num_states * horizon

    equality_constraints = problem.equality_constraints(opt_vars, problem.params)
    assert jnp.all(
        equality_constraints[: len(initial_constraints)] == initial_constraints
    )
    assert jnp.all(
        equality_constraints[len(initial_constraints) :] == dynamics_constraints
    )

    cost = problem.cost(opt_vars, problem.params)
    assert cost > 0


def test_optimal_control_problem():
    """Tests OptimalControlProblem."""
    dynamics = Dynamics()
    problem = OptimalControlProblem(dynamics=dynamics, parameters=problem_parameters)
    verify_problem(problem)


def test_feedback_optimal_control_problem():
    """Tests FeedbackOptimalControlProblem."""
    dynamics = FeedbackDynamics()
    problem = FeedbackOptimalControlProblem(
        dynamics=dynamics, parameters=problem_parameters
    )
    verify_problem(problem)
