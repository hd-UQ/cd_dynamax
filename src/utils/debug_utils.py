import jax.numpy as jnp
from jax import lax
import jax.debug as jdb


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
    sequence_length = length if length is not None else len([x for x in xs if x is not None][0])

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
