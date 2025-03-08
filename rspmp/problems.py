"""Optimal control roblems."""

from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np
from jax import vmap

from rspmp.dynamics import Dynamics

problem_parameters = {
    "state_quad_penalization_scalar": 10.0,
    "control_quad_penalization_scalar": 3.0,
    "discretization_time": 0.1,
    "sample_size": 10,
    "horizon": 20,
    "initial_state": np.zeros(3),  # num_states
    # samples of increments of Brownian motion
    "DWs": np.zeros((10, 20, 3)),  # (sample_size, horizon, num_states)
}


class OptimalControlProblem:
    """Optimal control problem.

    Optimization variables are stored in a vector
    optvars = (xs_vec, us_vec), where
    - xs_vec is the state trajectory and is of shape M*(N+1)*n_x
    - us_vec is the nominal control trajectory and is of shape N*n_u.
    """

    def __init__(self, dynamics: Dynamics, parameters: Dict[str, Any], verbose=False):
        """Initializes the class.

        Args:
            dynamics: dynamics
                (Dynamics) class
            parameters: problem parameters
                Dict[str, Any] dictionary
            verbose: verbose flag
                (bool)
        """
        if verbose:
            print("Initializing problem class.")
        self.dynamics = dynamics
        self.params = parameters
        self._discretization_time = self.params["discretization_time"]
        self._horizon = self.params["horizon"]
        self._sample_size = self.params["sample_size"]
        self._num_variables = int(
            self.sample_size * (self.horizon + 1) * self.dynamics.num_states
            + self.horizon * self.dynamics.num_controls
        )

    @property
    def dt(self) -> float:
        """Returns the discretization time."""
        return self._discretization_time

    @property
    def horizon(self) -> int:
        """Returns the horizon."""
        return self._horizon

    @property
    def sample_size(self) -> int:
        """Returns the sample size."""
        return self._sample_size

    @property
    def num_variables(self) -> int:
        """Returns the number of optimization variables."""
        return self._num_variables

    @property
    def dimensions(self) -> Tuple[int, int, int, int, int]:
        """Returns the problem dimensions."""
        return (
            self.sample_size,
            self.horizon,
            self.dynamics.num_states,
            self.dynamics.num_controls,
            self.num_variables,
        )

    @property
    def controls_start_index(self) -> int:
        """Returns the index in the optimization variables of the start of the controls.

        Optimization variables are in optvars = (xs_vec, us_vec) and
        xs_vec is of length sample_size * (horizon + 1) * n_x.
        """
        sample_size, horizon, n_x, _, _ = self.dimensions
        return sample_size * (horizon + 1) * n_x

    def convert_optvars_to_states_controls(self, optvars):
        """Returns states and controls in optvars.

        Args:
            optvars: optimization variables
                (num_variables) array

        Returns:
            state_matrix: states
                (sample_size, horizon+1, num_states) array
            control_matrix: controls
                (horizon, num_controls) array
        """
        M, N, n_x, n_u, _ = self.dimensions
        xs_vec = optvars[: self.controls_start_index]
        us_vec = optvars[self.controls_start_index :]
        xs = jnp.reshape(xs_vec, (n_x, M, N + 1), "F")
        us = jnp.reshape(us_vec, (n_u, N), "F")
        xs = jnp.moveaxis(xs, 0, -1)  # (M, N+1, n_x)
        us = us.T  # (N, n_u)
        return xs, us

    def convert_xs_us_mats_to_optvars(self, states, controls):
        """Packs states and controls in optvars.

        Args:
            state_matrix: states
                (sample_size, horizon+1, num_states) array
            control_matrix: controls
                (horizon, num_controls) array

        Returns:
            optvars: optimization variables
                (num_variables) array
        """
        states = jnp.moveaxis(states, 0, -1)  # (n_x, N+1, M)
        controls = jnp.moveaxis(controls, 0, -1)  # (n_u, N)
        states = jnp.reshape(states, (-1), "F")
        controls = jnp.reshape(controls, (-1), "F")
        optvars = jnp.concatenate([states, controls])
        return optvars

    def initial_guess_optvars(self):
        """Returns an initial guess.

        Returns:
            optvars: optimization variables
                (num_variables) array
        """
        num_variables = self.num_variables
        return 1e-6 * np.random.rand(num_variables)

    def initial_guess_initial_adjoints(self):
        """Returns an initial guess for the initial value of the adjoint vectors.

        Returns:
            initial_adjoints_guess: initial values of the adjoint vector
                (sample_size, num_states) array
        """
        M, _, n_x, _, _ = self.dimensions
        initial_adjoints_guess = 1e-6 * np.random.rand(M * n_x)
        initial_adjoints_guess = jnp.reshape(initial_adjoints_guess, (M, n_x))
        return initial_adjoints_guess

    def initial_constraints(self, optvars):
        """Returns the initial state constraints.

        Args:
            optvars: optimization variables
                (num_variables) array

        Returns:
            constraints: initial state constraints
                (sample_size * num_states) array
        """
        xs, _ = self.convert_optvars_to_states_controls(optvars)

        def initial_constraint_one_sample(xs_matrix):
            return xs_matrix[0] - self.params["initial_state"]

        eqs = vmap(initial_constraint_one_sample)(xs)
        return eqs.flatten()

    def dynamics_constraints(self, optvars, params):
        """Returns the dynamics constraints.

        Args:
            optvars: optimization variables
                (num_variables) array
            params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            constraints: dynamics constraints
                (sample_size * horizon * num_states) array
        """
        xs_mat, us_mat = self.convert_optvars_to_states_controls(optvars)

        def dynamics_constraint(x, u, xn, w):
            x_pred = x + self.dt * self.dynamics.b(x, u) + self.dynamics.sigma(x) @ w

            def milstein_term(x, dw):
                sig_diag = self.dynamics.sigma_diagonal_terms(x)
                sig_dx_diag = self.dynamics.sigma_dx_diagonal_terms(x)
                return 0.5 * sig_diag * sig_dx_diag * dw**2

            x_pred += milstein_term(x, w)
            return xn - x_pred

        Xs = xs_mat[:, :-1, :]
        Xns = xs_mat[:, 1:, :]
        Ws = params["DWs"]
        gs = vmap(vmap(dynamics_constraint), in_axes=(0, None, 0, 0))(
            Xs, us_mat, Xns, Ws
        )
        return gs.flatten()

    def equality_constraints(self, optvars, params):
        """Returns the equality constraints.

        Args:
            optvars: optimization variables
                (num_variables) array
            params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            constraints: equality constraints
                (sample_size * (horizon + 1) * num_states) array
        """
        initial_eq = self.initial_constraints(optvars)
        dynamics_eq = self.dynamics_constraints(optvars, params)
        return jnp.concatenate([initial_eq, dynamics_eq])

    def state_cost(self, state, params):
        """Returns the state cost.

        Args:
            state: state
                (num_states) array
            params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            cost: state cost
                (float)
        """
        return 0.5 * params["state_quad_penalization_scalar"] * jnp.sum(state**2)

    def control_cost(self, control, params):
        """Returns the control cost.

        Args:
            control: control input
                (num_controls) array
            params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            cost: control cost
                (float)
        """
        return 0.5 * params["control_quad_penalization_scalar"] * jnp.sum(control**2)

    def step_cost(self, state, control, params):
        """Returns the step cost.

        Args:
            state: state
                (num_states) array
            control: control input
                (num_controls) array
            params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            cost: state and control step cost
                (float)
        """
        return self.state_cost(state, params) + self.control_cost(control, params)

    def cost(self, optvars, params):
        """Returns the total cost over the horizon.

        Args:
            optvars: optimization variables
                (num_variables) array
            params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            cost: total cost
                (float)
        """
        xs_mat, us_mat = self.convert_optvars_to_states_controls(optvars)
        total_state_cost = jnp.sum(
            jnp.mean(
                vmap(vmap(self.state_cost, in_axes=(0, None)), in_axes=(0, None))(
                    xs_mat, params
                ),
                axis=0,
            )
        )
        total_control_cost = jnp.sum(
            vmap(self.control_cost, in_axes=(0, None))(us_mat, params)
        )
        total_cost = total_state_cost + total_control_cost
        total_cost = self.dt * total_cost
        return total_cost


class FeedbackOptimalControlProblem(OptimalControlProblem):
    """Feedback optimal control problem.

    Optimization variables are stored in a vector
    optvars = (xs_vec, us_vec), where
    - xs_vec is the state trajectory and is of shape M*(N+1)*n_x
    - us_vec is the nominal control trajectory and is of shape N*n_u.
    """

    def control_cost(self, state, control, params):
        """Returns the control cost.

        Args:
            state: state
                (num_states) array
            control: control input
                (num_controls) array
            params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            cost: control cost
                (float)
        """
        return (
            0.5
            * params["control_quad_penalization_scalar"]
            * jnp.sum((control * state) ** 2)
        )

    def step_cost(self, state, control, params):
        """Returns the step cost.

        Args:
            state: state
                (num_states) array
            control: control input
                (num_controls) array
            params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            cost: state and control step cost
                (float)
        """
        return self.state_cost(state, params) + self.control_cost(
            state, control, params
        )

    def cost(self, optvars, params):
        """Returns the total cost over the horizon.

        Args:
            optvars: optimization variables
                (num_variables) array
            params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            cost: total cost
                (float)
        """
        xs_mat, us_mat = self.convert_optvars_to_states_controls(optvars)
        total_state_cost = jnp.sum(
            jnp.mean(
                vmap(vmap(self.state_cost, in_axes=(0, None)), in_axes=(0, None))(
                    xs_mat, params
                ),
                axis=0,
            )
        )
        total_control_cost = jnp.sum(
            jnp.mean(
                vmap(
                    vmap(self.control_cost, in_axes=(0, 0, None)),
                    in_axes=(None, 0, None),
                )(us_mat, xs_mat[:, :-1], params),
                axis=0,
            )
        )
        total_cost = total_state_cost + total_control_cost
        total_cost = self.dt * total_cost
        return total_cost
