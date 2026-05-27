# JAX PSD check utility
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax
from typing import Optional, Tuple


def resolve_verbosity(warn: bool = True, verbosity: Optional[int] = None) -> Tuple[bool, int]:
    """Resolve the effective warning/debug verbosity.

    Verbosity levels:
        0: errors only
        1: warnings
        2: debug information

    If ``verbosity`` is omitted, preserve the historical ``warn`` behavior by
    mapping ``warn=True`` to level 1 and ``warn=False`` to level 0.
    """
    if verbosity is None:
        verbosity = 1 if warn else 0
    elif verbosity not in (0, 1, 2):
        raise ValueError(
            f"Invalid verbosity={verbosity}. Expected one of 0, 1, or 2."
        )

    return verbosity >= 1, verbosity


# PSD checks
def is_psd_eig(matrix, tol=1e-8):
    eigvals = jnp.linalg.eigvalsh(matrix)
    return jnp.all(eigvals >= -tol)


def is_psd_cholesky(matrix):
    # Cholesky only works for positive definite, not semi-definite
    # In JIT, exceptions are not catchable, so this is best-effort
    try:
        _ = jax.scipy.linalg.cholesky(matrix, lower=True)
        return True
    except Exception:
        return False


def psd(
    matrix,
    check_psd=True,
    psd_check_cholesky=False,
    tol=1e-8,
    warn=True,
    verbosity: Optional[int] = None,
    raise_error=False,
):
    """
    Try to return a psd matrix
        - first, by symmetrizing
        - then simply checking if it matrix is positive semi-definite (PSD).
            If not, optionally print a warning using jax.debug.print.
            TODO: raise jax error
    Args:
        matrix: array-like, matrix to check
        check_psd: bool, if True, check if the matrix is PSD
        psd_check_cholesky: bool, if True, use Cholesky decomposition to check for PSD
        tol: float, tolerance for eigenvalue PSD check
        warn: bool, if True print warning if not PSD
    Returns:
        matrix: array-like, hopefully PSD matrix
    """
    warn, verbosity = resolve_verbosity(warn=warn, verbosity=verbosity)

    def check_cholesky(mat):
        return is_psd_cholesky(mat)

    def check_eig(mat):
        return is_psd_eig(mat, tol)

    def symmetrize(mat):
        """Symmetrize one or more matrices."""
        return 0.5 * (mat + jnp.swapaxes(mat, -1, -2))

    def handle_not_psd(_):
        if warn:
            jax.debug.print("Warning: matrix is not positive definite")
        if raise_error:
            # TODO: raise jax error
            raise ValueError("Matrix is not positive definite")
        return None

    # Symmetrize the input matrix
    sym_matrix = symmetrize(matrix)

    # Check PSD
    if check_psd:
        is_psd = jax.lax.cond(psd_check_cholesky, check_cholesky, check_eig, sym_matrix)

        # TODO: better to move this logic out? namely, return is_psd, then check, so we can indicate where non_psd occurred?
        # Handle non_psd
        jax.lax.cond(~is_psd, handle_not_psd, lambda _: None, operand=None)

    # For now, just return the symmetrized matrix, regardless of PSD status
    return sym_matrix


# Wrapper over jax.lax.scan
def lax_scan(f, init, xs, length=None, reverse=False, debug=False):
    """
    A debugging wrapper around `lax.scan` that supports multiple input sequences, including None elements,
    and output sequences, with proper handling of reverse iteration.

    Parameters:
    - f: The function to apply. Signature: `f(carry, x) -> (carry, y)` where `x` and `y` can be tuples for multiple inputs/outputs.
    - init: The initial carry value.
    - xs: The input sequences to scan over. Can be a single array, a tuple of arrays, or None.
    - length: (Optional) The length of the input sequences. Required if xs is None or contains None elements.
    - reverse: (Optional) True to iterate in reverse order.
    - debug: If True, uses a for-loop for debugging. If False, uses `lax.scan`.

    Returns:
    - A tuple (carry, ys) where `carry` is the final carry value and `ys` are the scanned results.
    """

    if not debug:
        return lax.scan(f, init, xs, length=length, reverse=reverse)

    carry = init
    ys_lists = None

    # Ensure xs is a tuple for consistency, handle None by creating an appropriate placeholder
    xs = xs if isinstance(xs, tuple) else (xs,)
    sequence_length = (
        length if length is not None else len([x for x in xs if x is not None][0])
    )

    indices = range(sequence_length - 1, -1, -1) if reverse else range(sequence_length)

    for i in indices:
        # Construct the tuple for the current step, handling None elements appropriately
        x_i = tuple(x[i] if x is not None else None for x in xs)
        carry, y = f(carry, x_i)

        if y is None:
            continue

        # Initialize the ys_lists structure based on the type of y
        if ys_lists is None:
            if isinstance(y, dict):
                ys_lists = {key: [] for key in y.keys()}
            else:
                ys_lists = tuple([] for _ in range(len(y)))

        # Append the y components to the appropriate lists
        if isinstance(y, dict):
            for key, y_component in y.items():
                ys_lists[key].append(y_component)
        else:
            for list_index, y_component in enumerate(y):
                ys_lists[list_index].append(y_component)

    if ys_lists is None:
        ys = None
    else:
        if isinstance(ys_lists, dict):
            ys = {key: jnp.stack(y_list, axis=0) for key, y_list in ys_lists.items()}
        else:
            ys = tuple(jnp.stack(y_list, axis=0) for y_list in ys_lists)

    return carry, ys
