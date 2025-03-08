"""Utils."""

from functools import partial

import jax.numpy as jnp
from jax import jvp, vmap
from jax._src.api_util import check_callable


def value_and_jacfwd(f, x):
    """Returns the value and gradient of a function f evaluated at x."""
    check_callable(f)
    pushfwd = partial(jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = vmap(pushfwd, out_axes=(None, -1))((basis,))
    return y, jac


# def value_and_jacrev(f, x):
#     check_callable(f)
#     y, pullback = vjp(f, x)
#     basis = jnp.eye(y.size, dtype=y.dtype)
#     jac = jax.vmap(pullback)(basis)[0]
#     return y, jac
