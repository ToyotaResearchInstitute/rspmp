"""Tests dynamics."""

import jax.numpy as jnp

from rspmp.dynamics import Dynamics, FeedbackDynamics


def verify_dynamics(dynamics: Dynamics):
    """General checks for dynamics."""
    assert isinstance(dynamics, Dynamics)
    num_states = dynamics.num_states
    num_controls = dynamics.num_controls
    num_disturbances = dynamics.num_disturbances

    assert isinstance(num_states, int)
    assert isinstance(num_controls, int)
    assert isinstance(num_disturbances, int)
    assert num_states >= 1
    assert num_controls >= 1
    assert num_disturbances >= 1

    state = jnp.ones(num_states)
    control = jnp.ones(num_controls)
    disturbance = jnp.ones(num_disturbances)

    b = dynamics.b(state, control)
    assert isinstance(b, jnp.ndarray)
    assert len(b.shape) == 1
    assert b.shape[0] == num_states

    b1 = dynamics.b1(state)
    assert isinstance(b1, jnp.ndarray)
    assert len(b1.shape) == 1
    assert b1.shape[0] == num_states

    b2 = dynamics.b2(state)
    assert isinstance(b2, jnp.ndarray)
    assert len(b2.shape) == 2
    assert b2.shape[0] == num_states
    assert b2.shape[1] == num_controls

    sig = dynamics.sigma(state)
    assert isinstance(sig, jnp.ndarray)
    assert len(sig.shape) == 2
    assert sig.shape[0] == num_states
    assert sig.shape[1] == num_disturbances

    assert len(b2 @ control) == num_states
    assert len(sig @ disturbance) == num_states


def test_dynamics():
    """Tests Dynamics."""
    dynamics = Dynamics()
    verify_dynamics(dynamics)


def test_feedback_dynamics():
    """Tests FeedbackDynamics."""
    dynamics = FeedbackDynamics()
    verify_dynamics(dynamics)
