"""Dynamics classes."""

import jax.numpy as jnp


class Dynamics:
    """Stratonovich SDE dynamics dx = (b1(x) + b2(x)u)dt + sigma(x)dW."""

    def __init__(self, verbose=False):
        """Initializes the class."""
        if verbose:
            print("Initializing dynamics class.")
        self._num_states = 3
        self._num_controls = 3
        self._num_disturbances = 3
        inertia_matrix = jnp.diag(jnp.array([3.0, 2.0, 4.0]))
        self.inertia = inertia_matrix
        self.inertia_inverse = jnp.linalg.inv(inertia_matrix)
        self.diffusion_magnitude = 0.4

    @property
    def num_states(self) -> int:
        """Returns the number of state variables."""
        return self._num_states

    @property
    def num_controls(self) -> int:
        """Returns the number of control variables."""
        return self._num_controls

    @property
    def num_disturbances(self) -> int:
        """Returns the number of disturbance variables."""
        return self._num_disturbances

    def b(self, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        """Returns the total drift term.

        Returns the total drift term
        b(x,u) = (b1(x) + b2(x)u) of the Stratonovich SDE dynamics
        dx = (b1(x) + b2(x)u)dt + sigma(x)dW.

        Args:
            state: state of the system (x variable)
                (num_states) array
            control: control of the system (x variable)
                (num_controls) array

        Returns:
            b_value: value b(x,u)
                (num_states) array
        """
        bval = self.b1(state) + self.b2(state) @ control
        return bval

    def b1(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns the uncontrolled drift term.

        Returns the drift term b1(x) of the Stratonovich SDE dynamics
        dx = (b1(x) + b2(x)u)dt + sigma(x)dW.

        Args:
            state: state of the system (x variable)
                (num_states) array

        Returns:
            b1_value: value b1(x)
                (num_states) array
        """
        omega = state
        ox, oy, oz = omega
        omega_cross = jnp.array([[0, -oz, oy], [oz, 0, -ox], [-oy, ox, 0]])
        omega_dot = self.inertia_inverse @ (-omega_cross @ self.inertia @ omega)
        return omega_dot

    def b2(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns the controlled drift term.

        Returns the drift term b2(x) of the Stratonovich SDE dynamics
        dx = (b1(x) + b2(x)u)dt + sigma(x)dW.

        Args:
            state: state of the system (x variable)
                (num_states) array

        Returns:
            b2_value: value b2(x)
                (num_states, num_controls) array
        """
        return self.inertia_inverse

    def sigma(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns the diffusion matrix.

        Returns the diffusion term sigma(x) of the Stratonovich SDE dynamics
        dx = (b1(x) + b2(x)u)dt + sigma(x)dW.

        Args:
            state: state of the system (x variable)
                (num_states) array

        Returns:
            sigma_value: matrix sigma(x)
                (num_states, num_states) array
        """
        sig = self.diffusion_magnitude * jnp.diag(state)
        return sig

    def sigma_diagonal_terms(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns the diagonal terms of the matrix sigma(x).

        Args:
            state: state of the system (x variable)
                (num_states) array

        Returns:
            value: diagonal terms of the matrix sigma(x)
                (num_states) array
        """
        return self.diffusion_magnitude * state

    def sigma_dx_diagonal_terms(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns the diagonal terms of the tensor ∇σ(x).

        Args:
            state: state of the system (x variable)
                (num_states) array

        Returns:
            value: diagonal terms of the tensor ∇σ(x)
                (num_states) array
        """
        return self.diffusion_magnitude * jnp.ones_like(state)

    def sigma_ddx_diagonal_terms(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns the diagonal terms of the tensor ∇σ(x)σ(x).

        Args:
            state: state of the system (x variable)
                (num_states) array

        Returns:
            value: diagonal terms of the tensor ∇σ(x)σ(x)
                (num_states) array
        """
        return jnp.zeros_like(state)


class FeedbackDynamics(Dynamics):
    """
    Stratonovich dynamics dx = (b1(x) + b2(x)u)dt + sigma(x)dW
    corresponding to the closed loop dynamics of Dynamics.

    If b1(x) and b2(x) are the vector fields of Dynamics,
    FeedbackDynamics has the dynamics
    dx = (b1(x) + b2(x)@diag(x)@u)dt + sigma(x)dW
       = (b1(x) + b2feedback(x)@u)dt + sigma(x)dW.

    This class assumes that the b2(x) of the base class is diagonal,
    and the feedback gains u correspond to a diagonal feedback gain matrix
    K = diag(u), so that the control term is
    b2(x) @ (K@x) = (diag(b2(x)) @ diag(x)) @ u.
    """

    def b2(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns the controlled drift term.

        Returns the drift term b2(x) of the dynamics
        dx = (b1(x) + b2(x)u)dt + sigma(x)dW.

        Args:
            state: state of the system (x variable)
                (num_states) array

        Returns:
            b2_value: value b2(x)
                (num_states, num_controls) array
        """
        return jnp.diag(jnp.diag(self.inertia_inverse) * state)
