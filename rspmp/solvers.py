"""Direct and indirect solvers for optimal control."""

from functools import partial
from time import time
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
import osqp
from jax import config, grad, hessian, jacfwd, jit, vmap
from jax.lax import scan
from scipy.sparse import csc_matrix

from rspmp.problems import FeedbackOptimalControlProblem, OptimalControlProblem
from rspmp.utils import value_and_jacfwd

config.update("jax_debug_nans", True)

solver_parameters = {
    "num_iterations_max": 10,
    "tolerance": 1e-6,
    "verbose": False,
    "osqp_verbose": False,
    "osqp_tolerance": 1e-6,
    "osqp_polish": True,
    "warm_start": True,
}


class SQPSolver:
    """Sequential quadratic programming solver.

    Internally, optimization variables are stored in a vector
    optvars = (xs_vec, us_vec), where
    - xs_vec is the state trajectory and is of shape M*(N+1)*n_x
    - us_vec is the nominal control trajectory and is of shape N*n_u.
    """

    def __init__(
        self, problem: OptimalControlProblem, parameters: Dict[str, Any], verbose=False
    ):
        """Initializes the class.

        Args:
            problem: optimal control problem
                (OptimalControlProblem) class
            parameters: solver parameters
                Dict[str, Any] dictionary
            verbose: verbose flag
                (bool)
        """
        if isinstance(problem, FeedbackOptimalControlProblem):
            msg = "This solver does not support non-convex objectives."
            raise NotImplementedError(msg)
        if verbose:
            print("Initializing solver class.")
        self.params = parameters
        self.problem = problem

        if verbose:
            print(">>> SQPSolver: Pre-compiling")
        optvars = self.problem.initial_guess_optvars()
        cost_hessian = csc_matrix(self.cost_hessian(problem.params))
        constraints_jacobian = csc_matrix(
            self.equality_constraints_dz(optvars, problem.params)
        )
        self.cost_hessian_nonzero_indices = cost_hessian.nonzero()
        self.constraints_jacobian_nonzero_indices = constraints_jacobian.nonzero()
        self.num_equality_constraints = len(
            self.problem.equality_constraints(optvars, problem.params)
        )
        self.cost_hessian_nonzero_entries(problem.params)
        self.equality_constraints_osqp_matrices_nonzero_entries(optvars, problem.params)

        num_iter = self.params["num_iterations_max"]
        self.params["num_iterations_max"] = 2
        self.solve(verbose=False)
        self.params["num_iterations_max"] = num_iter
        if verbose:
            print(">>> SQPSolver: Pre-compiling: Done")

    @partial(jit, static_argnums=(0,))
    def cost_hessian(self, problem_params):
        """Returns the hessian of the cost.

        Args:
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            hessian: hessian of the cost
                (num_variables, num_variables) array
        """
        hess = hessian(self.problem.cost)(
            np.zeros(self.problem.num_variables), problem_params
        )
        return hess

    @partial(jit, static_argnums=(0,))
    def equality_constraints_dz(self, optvars, problem_params):
        """Returns the jacobian dh/dz(z) of the constraints h(z)=0.

        Args:
            optvars: optimization variables
                (num_variables) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            jacobian: equality constraints jacobian
                (num_constraints, num_variables) array
        """
        return jacfwd(self.problem.equality_constraints)(optvars, problem_params)

    @partial(jit, static_argnums=(0,))
    def cost_hessian_nonzero_entries(self, problem_params):
        """Returns the nonzero terms of cost hessian.

        Args:
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            hessian_nonzero_entries: nonzero entries of the hessian of the cost
                (num_nonzero_entries) array
        """
        return self.cost_hessian(problem_params)[self.cost_hessian_nonzero_indices]

    # --------------------------------------------------------------------
    # OSQP helper functions.
    #
    # OSQP (https://osqp.org/) solves quadratic programs (QP) of the form
    #     min_z       0.5 z^T P z
    #     such that   l <= A z <= u.
    # --------------------------------------------------------------------

    @partial(jit, static_argnums=(0,))
    def equality_constraints_osqp_matrices(self, optvars, problem_params):
        """Returns the terms (A,l,u) of the constraints l <= Az <= u.

        Args:
            optvars: optimization variables
                (num_variables) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            A: matrix A
                (num_constraints, num_variables) array
            l: lower bound l
                (num_constraints) array
            u: upper bound u
                (num_constraints) array
        """
        h, dh_dz = value_and_jacfwd(
            lambda z: self.problem.equality_constraints(z, problem_params), optvars
        )
        # h(z) = 0
        # => h(zp) + dh/dz(zp) @ (z - zp) = 0
        # => dh/dz(zp) @ z = -(h(zp) - dh/dz(zp) @ zp)
        Aeq = dh_dz
        leq = -h + dh_dz @ optvars
        ueq = leq
        return Aeq, leq, ueq

    @partial(jit, static_argnums=(0,))
    def equality_constraints_osqp_matrices_nonzero_entries(
        self, optvars, problem_params
    ):
        """Returns the nonzero terms (A,l,u) of the constraints l <= Az <= u.

        Args:
            optvars: optimization variables
                (num_variables) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            A: nonzero entries of the matrix A
                (num_nonzero_entries) array
            l: lower bound l
                (num_constraints) array
            u: upper bound u
                (num_constraints) array
        """
        Aeq, leq, ueq = self.equality_constraints_osqp_matrices(optvars, problem_params)
        Aeq = Aeq[self.constraints_jacobian_nonzero_indices]
        return Aeq, leq, ueq

    def get_constraints_coeffs(self, optvars, problem_params):
        """Computes the constraints l <= Az <= u of the quadratic program.

        Args:
            optvars: optimization variables
                (num_variables) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            A: matrix A
                (num_constraints, num_variables) scripy matrix
            l: lower bound l
                (num_constraints) array
            u: upper bound u
                (num_constraints) array
        """
        num_vars = self.problem.num_variables
        A, lower_bound, upper_bound = (
            self.equality_constraints_osqp_matrices_nonzero_entries(
                optvars, problem_params
            )
        )
        A, lower_bound, upper_bound = (
            np.array(A),
            np.array(lower_bound),
            np.array(upper_bound),
        )
        A = csc_matrix(
            (A, self.constraints_jacobian_nonzero_indices),
            shape=(self.num_equality_constraints, num_vars),
        )
        return A, lower_bound, upper_bound

    def get_objective_coeffs(self, problem_params):
        """Computes the coefficients of (1/2 z^T P z + q^T z) of the quadratic program.

        Args:
            optvars: optimization variables
                (num_variables) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            P: nonzero entries of the matrix P
                (num_nonzero_entries, num_nonzero_entries) scipy matrix
            q: vector q
                (num_variables) array
        """
        # Returns (P, q) corresponding to objective
        #        min (1/2 z^T P z + q^T z)
        # where z is the optimization variable.
        num_vars = self.problem.num_variables
        # Quadratic Objective
        P = np.array(self.cost_hessian_nonzero_entries(problem_params))
        P = 2.0 * P  # osqp cost is actually 1/2 z^T P z
        P = csc_matrix(
            (P, self.cost_hessian_nonzero_indices), shape=(num_vars, num_vars)
        )
        # Linear Objective
        q = np.zeros(num_vars)
        return P, q

    def define_qp(self, optvars, problem_params):
        """Defines the quadratic program."""
        # objective and constraints
        self.P, self.q = self.get_objective_coeffs(problem_params)
        self.A, self.l, self.u = self.get_constraints_coeffs(optvars, problem_params)
        # Setup OSQP problem
        self.osqp_prob = osqp.OSQP()
        self.osqp_prob.setup(
            self.P,
            self.q,
            self.A,
            self.l,
            self.u,
            linsys_solver="qdldl",
            warm_start=self.params["warm_start"],
            verbose=self.params["osqp_verbose"],
            eps_abs=self.params["osqp_tolerance"],
            eps_rel=self.params["osqp_tolerance"],
        )
        return True

    def solve_qp(self):
        """Solves the quadratic program.

        Returns:
            optvars: solution of the QP
                (num_variables) array
            kkt_multipliers: KKT multipliers of the constraints of the QP
                (num_constraints) array
        """
        self.res = self.osqp_prob.solve()
        if self.res.info.status != "solved":
            print("[solve]: Problem infeasible.")
        return self.res.x, self.res.y

    @partial(jit, static_argnums=(0,))
    def error(self, optvars, optvars_prev):
        """Computes the convergence error.

        Args:
            optvars: optimization variables
                (num_variables) array
            optvars_prev: previous optimization variables
                (num_variables) array

        Returns:
            convergence_error: infinity norm ||optvars-optvars_prev||_infty
                (float)
        """
        return jnp.linalg.norm(optvars - optvars_prev, ord=np.inf)

    def solve(self, initial_guess=None, problem_params=None, verbose=False):
        """Solves the problem.

        Args:
            initial_guess: optimization variables initial guess
                (num_variables) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary
            verbose: verbose flag
                (bool)

        Returns:
            state_matrix: states
                (sample_size, horizon+1, num_states) array
            control_matrix: controls
                (horizon, num_controls) array
            initial_adjoints: initial values of the adjoint vector
                (sample_size, num_states) array
            convergence_error: final convergence error
                (float)
            solve_time: total computation time
                (float)
        """
        if verbose:
            print("SQPSolver::solve()")
        M, _, n_x, n_u, num_vars = self.problem.dimensions
        if initial_guess is None:
            initial_guess = self.problem.initial_guess_optvars()
        if problem_params is None:
            problem_params = self.problem.params
        optvars_prev = initial_guess

        start_time = time()
        total_define_time = 0
        total_solve_time = 0
        for scp_iter in range(self.params["num_iterations_max"]):
            define_time = time()
            self.define_qp(optvars_prev, problem_params)
            total_define_time += time() - define_time
            solve_time = time()
            optvars, y = self.solve_qp()
            total_solve_time += time() - solve_time
            convergence_error = self.error(optvars, optvars_prev)
            if verbose:
                print("scp_iter =", scp_iter, "error =", convergence_error)
            optvars_prev = optvars
            if convergence_error < self.params["tolerance"]:
                break
        solve_time = time() - start_time
        if verbose:
            print("Total elapsed = ", solve_time)
            print(">> defining: ", total_define_time)
            print(">> solving: ", total_solve_time)

        initial_adjoints = jnp.reshape(y[: M * n_x], (n_x, M), "F").T
        states, controls = self.problem.convert_optvars_to_states_controls(optvars)
        return states, controls, initial_adjoints, convergence_error, solve_time


class IndirectSolver:
    """Indirect shooting method solver.

    The solver searches for the initial adjoint (costate) vectors such that
    the conditions of the Pontryagin Maximum Principle are satisfied, using
    a Newton method.
    """

    def __init__(
        self, problem: OptimalControlProblem, parameters: Dict[str, Any], verbose=False
    ):
        """Initializes the class.

        Args:
            problem: optimal control problem
                (OptimalControlProblem) class
            parameters: solver parameters
                Dict[str, Any] dictionary
            verbose: verbose flag
                (bool)
        """
        if isinstance(problem, FeedbackOptimalControlProblem):
            msg = "This solver does not support feedback optimization."
            raise NotImplementedError(msg)
        if verbose:
            print("Initializing solver class.")
        self.params = parameters
        self.problem = problem
        try:
            self.solve(verbose=False)
        except Exception as e:
            raise e
        if verbose:
            print(">>> Done")

    def hamiltonian(self, state, costate, control, problem_params):
        """Returns the Hamiltonian H(x,u,p).

        Args:
            state: state
                (num_states) array
            costate: costate (adjoint vector)
                (num_states) array
            control: control input
                (num_controls) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            hamiltonian: Hamiltonian H(x,u,p)
                (float)
        """
        return (
            -self.problem.step_cost(state, control, problem_params)
            + self.problem.dynamics.b(state, control).T @ costate
        )

    def optimal_control(self, states, costates, problem_params):
        """Returns the optimal control described by maximality condition of the PMP.

        Args:
            states: states
                (num_samples, num_states) array
            costates: costates (adjoint vectors)
                (num_samples, num_states) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            optimal_control: optimal control input
                (num_controls) array
        """
        R = problem_params["control_quad_penalization_scalar"]
        J_inv = jnp.diag(self.problem.dynamics.inertia_inverse)
        control = (J_inv / R) * jnp.mean(costates, axis=0)
        return control

    def adjoint_drift(self, state, costate, control, problem_params):
        """Drift of the adjoint vector from the adjoint equation of the PMP.

        Args:
            costate: state
                (num_states) array
            costate: costate (adjoint vector)
                (num_states) array
            control: control input
                (num_controls) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            adjoint_drift: drift of the adjoint vector
                (num_states) array
        """
        dH_dx = grad(self.hamiltonian)(state, costate, control, problem_params)
        return -dH_dx

    def adjoint_diffusion(self, state, costate):
        """Diffusion of the adjoint vector from the adjoint equation of the PMP.

        Args:
            costate: state
                (num_states) array
            costate: costate (adjoint vector)
                (num_states) array

        Returns:
            adjoint_diffusion: diffusion of the adjoint vector
                (num_states, num_states) array
        """
        sig_dx = jacfwd(lambda x: self.problem.dynamics.sigma(x))(state)
        sigma = -jnp.einsum("kji,k->ji", sig_dx, costate).T
        return sigma

    def shooting_drift(self, state_costate, control, problem_params):
        """Drift of the concatenated state-adjoint vector.

        Args:
            state_costate: state and costate
                (2*num_states) array
            control: control input
                (num_controls) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            shooting_drift: drift of the state-adjoint vector
                (2*num_states) array
        """
        # state_costate - (2 * n_x)
        state, costate = jnp.split(state_costate, 2)
        return jnp.concatenate(
            [
                self.problem.dynamics.b(state, control),
                self.adjoint_drift(state, costate, control, problem_params),
            ]
        )

    def shooting_diffusion(self, state_costate):
        """Diffusion of the concatenated state-adjoint vector.

        Args:
            state_costate: state and costate
                (2*num_states) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            shooting_diffusion: drift of the state-adjoint vector
                (2*num_states, num_states) array
        """
        state, costate = jnp.split(state_costate, 2)
        return jnp.concatenate(
            [self.problem.dynamics.sigma(state), self.adjoint_diffusion(state, costate)]
        )

    @partial(jit, static_argnums=(0,))
    def state_and_costate_trajectories(self, initial_costates, problem_params):
        """Trajectories of the states and costates for fixed initial adjoint vectors.

        Args:
            initial_costates: initial values of the adjoint vectors (costate)
                (sample_size, num_states) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            times: times
                (horizon+1) array
            states: state trajectory
                (sample_size, horizon+1, num_states) array
            costates: costate trajectory (adjoint)
                (sample_size, horizon+1, num_states) array
        """
        dt = self.problem.dt
        M, N, _, _, _ = self.problem.dimensions

        def next_scan(states_costates, dws):
            # states_costates - (M, 2*n_x)
            # dws - (M, n_x)
            states, costates = jnp.split(states_costates, 2, axis=-1)
            control = self.optimal_control(states, costates, problem_params)

            shooting_drifts = vmap(self.shooting_drift, in_axes=(0, None, None))
            next_states_costates = states_costates + dt * shooting_drifts(
                states_costates, control, problem_params
            )

            def diffusion_term(state_costate, dw):
                return self.shooting_diffusion(state_costate) @ dw

            next_states_costates += vmap(diffusion_term)(states_costates, dws)

            def milstein_term(state_costate, dw):
                state, costate = jnp.split(state_costate, 2)
                sig_diag = self.problem.dynamics.sigma_diagonal_terms(state)
                sig_dx_diag = self.problem.dynamics.sigma_dx_diagonal_terms(state)
                sig_ddx_diag = self.problem.dynamics.sigma_ddx_diagonal_terms(state)
                return 0.5 * jnp.concatenate(
                    [
                        sig_diag * sig_dx_diag * dw**2,
                        -(sig_ddx_diag * sig_diag + sig_dx_diag**2)
                        * costate
                        * dw**2,
                    ]
                )

            next_states_costates += vmap(milstein_term)(states_costates, dws)
            return next_states_costates, next_states_costates

        initial_states = jnp.repeat(
            self.problem.params["initial_state"][None], axis=0, repeats=M
        )
        initial_states_costates = jnp.concatenate(
            [initial_states, initial_costates], axis=1
        )
        dws = jnp.moveaxis(problem_params["DWs"], 0, 1)  # (N, M, n_x)

        _, states_costates = scan(next_scan, initial_states_costates, dws)

        states_costates = jnp.moveaxis(states_costates, 0, 1)  # (M, N, n_x)

        times = dt * jnp.arange(N + 1)  # of size (num_steps + 1)
        states_costates = jnp.concatenate(
            [initial_states_costates[:, jnp.newaxis, :], states_costates], axis=1
        )  # of size (M, num_steps + 1, 6)
        states, costates = jnp.split(states_costates, 2, axis=-1)
        return times, states, costates

    @partial(jit, static_argnums=(0,))
    def times_states_controls_costates(self, initial_costates, problem_params):
        """Trajectories of the states, controls, and costates for fixed initial adjoint vectors.

        Args:
            initial_costates: initial values of the adjoint vectors (costate)
                (sample_size, num_states) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            times: times
                (horizon+1) array
            states: state trajectory
                (sample_size, horizon+1, num_states) array
            controls: control trajectory
                (horizon, num_controls) array
            costates: costate trajectory (adjoint)
                (sample_size, horizon+1, num_states) array
        """
        times, states, costates = self.state_and_costate_trajectories(
            initial_costates, problem_params
        )
        controls = vmap(self.optimal_control, in_axes=(1, 1, None))(
            states, costates, problem_params
        )
        controls = controls[:-1]
        return times, states, controls, costates

    @partial(jit, static_argnums=(0,))
    def shooting_residual(self, initial_costates_flattened, problem_params):
        """Residual error for the transversality condition of the PMP.

        Args:
            initial_costates: initial values of the adjoint vectors (costate)
                (sample_size * num_states) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            residual: residual
                (sample_size * num_states) array
        """
        M, _, n_x, _, _ = self.problem.dimensions
        initial_costates = jnp.reshape(initial_costates_flattened, (M, n_x))
        _, _, costate_trajectories = self.state_and_costate_trajectories(
            initial_costates, problem_params
        )
        final_costates = costate_trajectories[:, -1]
        residual = final_costates.flatten()
        return residual

    @partial(jit, static_argnums=(0,))
    def shooting_residual_and_jacobian(self, initial_costates, problem_params):
        """Residual error and its Jacobian for the transversality condition of the PMP.

        Args:
            initial_costates: initial values of the adjoint vectors (costate)
                (sample_size * num_states) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            residual: residual
                (sample_size * num_states) array
            residual_dp: residual jacobian with respect to initial costates
                (sample_size * num_states, sample_size * num_states) array
        """
        # initial_costates - (M, n_x)
        M, N, n_x, n_u, num_vars = self.problem.dimensions
        initial_costates = jnp.reshape(initial_costates, (-1))
        residual, jac = value_and_jacfwd(
            lambda p0s: self.shooting_residual(p0s, problem_params), initial_costates
        )
        jac = jnp.reshape(jac, (M * n_x, M * n_x))
        return residual, jac

    @partial(jit, static_argnums=(0,))
    def newton_step(self, initial_costates, problem_params):
        """Performs one step of a Newton method to satisfy the PMP transversality condition.

        Args:
            initial_costates: initial values of the adjoint vectors (costate)
                (sample_size * num_states) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            residual: residual
                (sample_size * num_states) array
            costates: new adjoint vectors (costate)
                (sample_size * num_states) array
        """
        M, _, n_x, _, _ = self.problem.dimensions
        residual, jac = self.shooting_residual_and_jacobian(
            initial_costates, problem_params
        )
        delta_initial_costates = jnp.linalg.solve(jac, residual)
        delta_initial_costates = jnp.reshape(delta_initial_costates, (M, n_x), "C")
        return initial_costates - delta_initial_costates, residual

    @partial(jit, static_argnums=(0,))
    def error(self, optvars, optvars_prev):
        """Computes the convergence error.

        Args:
            optvars: optimization variables
                (num_variables) array
            optvars_prev: previous optimization variables
                (num_variables) array

        Returns:
            convergence_error: infinity norm ||optvars-optvars_prev||_infty
                (float)
        """
        return jnp.linalg.norm(optvars - optvars_prev, ord=np.inf)

    # @partial(jit, static_argnums=(0,))
    def solve(self, initial_costates_guess=None, problem_params=None, verbose=False):
        """Solves the optimal control problem via a indirect shooting method.

        Args:
            initial_costates_guess: initial values of the adjoint vectors (costate)
                (sample_size * num_states) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary
            verbose: verbose flag
                (bool)

        Returns:
            state_matrix: states
                (sample_size, horizon+1, num_states) array
            control_matrix: controls
                (horizon, num_controls) array
            initial_adjoints: initial values of the adjoint vector
                (sample_size, num_states) array
            convergence_error: final convergence error
                (float)
            solve_time: total computation time
                (float)
        """
        if verbose:
            print("IndirectSolver::solve()")
        if initial_costates_guess is None:
            initial_costates_guess = self.problem.initial_guess_initial_adjoints()
        if problem_params is None:
            problem_params = self.problem.params

        start_time = time()
        costates = initial_costates_guess
        error = -1.0
        for step in range(self.params["num_iterations_max"]):
            costates, residual = self.newton_step(costates, problem_params)
            error = self.error(residual, 0)
            if verbose:
                print("iter =", step, ", error =", error)
            if error < self.params["tolerance"]:
                break
        solve_time = time() - start_time
        if verbose:
            print("Total elapsed = ", time() - start_time)

        initial_adjoints = costates
        convergence_error = error
        _, states, controls, _ = self.times_states_controls_costates(
            initial_adjoints, problem_params
        )
        return states, controls, initial_adjoints, convergence_error, solve_time


class IndirectFeedbackSolver(IndirectSolver):
    """Indirect shooting method solver."""

    def __init__(
        self,
        problem: FeedbackOptimalControlProblem,
        parameters: Dict[str, Any],
        verbose=False,
    ):
        """Initializes the class.

        Args:
            problem: optimal control problem
                (FeedbackOptimalControlProblem) class
            parameters: solver parameters
                Dict[str, Any] dictionary
            verbose: verbose flag
                (bool)
        """
        if not isinstance(problem, FeedbackOptimalControlProblem):
            msg = "This solver does not support solving problem " + str(problem)
            raise NotImplementedError(msg)
        if verbose:
            print("Initializing solver class.")
        self.params = parameters
        self.problem = problem
        try:
            self.solve(verbose=False)
        except Exception as e:
            raise e
        if verbose:
            print(">>> Done")

    def optimal_control(self, states, costates, problem_params):
        """Returns the optimal control described by maximality condition of the PMP.

        Args:
            states: states
                (num_samples, num_states) array
            costates: costates (adjoint vectors)
                (num_samples, num_states) array
            problem_params: problem parameters
                Dict[str, jnp.array] dictionary

        Returns:
            optimal_control: optimal control input
                (num_controls) array
        """
        R = problem_params["control_quad_penalization_scalar"]
        ExxT = jnp.mean(states**2, axis=0)
        EpxT = jnp.mean(states * costates, axis=0)
        control = (jnp.diag(self.problem.dynamics.inertia_inverse) / R) * (EpxT / ExxT)
        return control
