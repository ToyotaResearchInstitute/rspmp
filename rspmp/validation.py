"""Validation classes."""

import copy
from functools import partial
from typing import Any, Dict, Type

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.lax import scan

from rspmp.dynamics import Dynamics
from rspmp.problems import OptimalControlProblem

validation_parameters = {
    "sample_size": 10000,
}


class MonteCarloOptimalControlValidation:
    """Validation of solutions to optimal control problems."""

    def __init__(
        self,
        problem_type: Type[OptimalControlProblem],
        dynamics: Dynamics,
        problem_parameters: Dict[str, Any],
        parameters: Dict[str, Any] = validation_parameters,
    ):
        """Initializes the class.

        Args:
            problem_type: type of optimal control problem
                (type)
            dynamics: dynamics
                (Dynamics) class
            problem_parameters: problem parameters
                Dict[str, Any] dictionary
            parameters: validation parameters
                Dict[str, Any] dictionary
        """
        self._dynamics = dynamics
        self._params = parameters

        problem_parameters = copy.deepcopy(problem_parameters)

        self._discretization_time = problem_parameters["discretization_time"]
        self._horizon = problem_parameters["horizon"]
        problem_parameters["sample_size"] = self.sample_size
        problem_parameters["DWs"] = self.sample_disturbances()

        self._problem_params = problem_parameters
        self._problem = problem_type(dynamics, problem_parameters, verbose=False)

    @property
    def dt(self) -> float:
        """Returns the discretization time."""
        return self._discretization_time

    @property
    def horizon(self) -> int:
        """Returns the horizon."""
        return self._horizon

    @property
    def sample_size(self) -> float:
        """Returns the sample size."""
        return self.params["sample_size"]

    @property
    def params(self) -> Dict:
        """Returns the parameters."""
        return self._params

    @property
    def problem_params(self) -> Dict:
        """Returns the problem parameters."""
        return self._problem_params

    @property
    def dynamics(self) -> Dynamics:
        """Returns the dynamics."""
        return self._dynamics

    @property
    def problem(self) -> OptimalControlProblem:
        """Returns the problem."""
        return self._problem

    def sample_disturbances(self) -> np.ndarray:
        """Samples disturbances."""
        DWs = np.random.randn(self.sample_size, self.horizon, self.dynamics.num_states)
        DWs = np.sqrt(self.dt) * DWs
        return DWs

    @partial(jit, static_argnums=(0,))
    def get_state_trajectories(self, control_trajectory, DWs):
        """Trajectories of the states and costates for fixed initial adjoint vectors.

        Args:
            control_trajectory: control inputs
                (horizon, num_controls) array
            DWs: sample of disturbances
                (sample_size, horizon, num_states) array

        Returns:
            states: state trajectory
                (sample_size, horizon+1, num_states) array
        """
        # dws - (M, N, n_x)
        ocp = self.problem
        dt = ocp.dt

        def next_scan(x, d):
            u, w = d["controls"], d["dws"]
            next_state = x + dt * ocp.dynamics.b(x, u) + ocp.dynamics.sigma(x) @ w

            def milstein_term(x, w):
                sig_diag = ocp.dynamics.sigma_diagonal_terms(x)
                sig_dx_diag = ocp.dynamics.sigma_dx_diagonal_terms(x)
                return 0.5 * sig_diag * sig_dx_diag * w**2

            next_state += milstein_term(x, w)
            return next_state, next_state

        def get_trajectory(initial_state, control_trajectory, dws):
            d = {"dws": dws, "controls": control_trajectory}
            _, states = scan(next_scan, initial_state, d)
            states = jnp.concatenate([initial_state[jnp.newaxis, :], states], axis=0)
            return states

        states = vmap(get_trajectory, in_axes=(None, None, 0))(
            ocp.params["initial_state"], control_trajectory, DWs
        )
        return states

    def get_solution_cost(self, control_trajectory):
        """Cost associated to the given control trajectory.

        Args:
            control_trajectory: control inputs
                (horizon, num_controls) array

        Returns:
            cost: cost associated to the given control trajectory
                (float)
        """
        ocp = self.problem
        DWs = self.sample_disturbances()
        state_trajectories = self.get_state_trajectories(control_trajectory, DWs)
        total_state_cost = jnp.sum(
            jnp.mean(
                vmap(vmap(ocp.state_cost, in_axes=(0, None)), in_axes=(0, None))(
                    state_trajectories, ocp.params
                ),
                axis=0,
            )
        )
        total_control_cost = jnp.sum(
            vmap(ocp.control_cost, in_axes=(0, None))(control_trajectory, ocp.params)
        )
        total_cost = ocp.dt * (total_state_cost + total_control_cost)
        return total_cost
