import jax.numpy as jnp
from jax import vmap
  
# Aux functions for tests
def try_all_close(x, y, start_tol=-8, end_tol=-4):
    """Try all close with increasing tolerance"""
    # create list of tols 1e-8, 1e-7, 1e-6, ..., 1e1
    tol_list = jnp.array([10 ** i for i in range(start_tol, end_tol+1)])
    for tol in tol_list:
        if jnp.allclose(x, y, atol=tol):
            return True, tol
    return False, tol

def compare(x, x_ref, do_det=False, accept_failure=False):
    allclose, tol = try_all_close(x, x_ref)
    if allclose:
        print(f"\tAllclose passed with atol={tol}.")
    else:
        print(f"\tAllclose FAILED with atol={tol}.")

        # if x is 1d, add batch dim
        # TODO: use ensure_array_has_batch_dim instead
        if x.ndim == 1:
            x = x[:, None]
            x_ref = x_ref[:, None]

        # compute MSE of determinants over time
        if do_det:
            x = vmap(jnp.linalg.det)(x)
            x_ref = vmap(jnp.linalg.det)(x_ref)
            mse = (x - x_ref) ** 2
            rel_mse = mse / (x_ref**2)
        else:
            mse = jnp.mean((x - x_ref) ** 2, axis=1)
            rel_mse = mse / jnp.mean(x_ref ** 2, axis=1)

        print("\tInitial relative MSE: ", rel_mse[0])
        print("\tFinal relative MSE: ", rel_mse[-1])
        print("\tMax relative MSE: ", jnp.max(rel_mse))
        print("\tAverage relative MSE: ", jnp.mean(rel_mse))

        allclose, tol = try_all_close(rel_mse, 0, end_tol=-3)
        if not accept_failure:
            assert allclose, f"Relative MSE allclose FAILED with atol={tol}. UNACCEPTABLE!"
        else:
            print(f"Relative MSE allclose FAILED with atol={tol} but accepting this failure.")

    pass


def is_namedtuple(instance):
    """Check if an instance is a namedtuple."""
    return isinstance(instance, tuple) and hasattr(instance, "_fields")


def compare_new(array1, array2, atol=1e-5):
    """Compare two arrays and return a tuple indicating if they are close and the comparison result."""
    are_close = jnp.allclose(array1, array2, atol=atol)
    return are_close, "same" if are_close else "different"


def get_array(data):
    """Attempt to extract a JAX array from the input or return it directly if already an array."""
    if isinstance(data, jnp.ndarray):
        return data
    else:
        try:
            return data.params
        except:
            return data


def compare_leaves(node1, node2, path="", atol=1e-5):
    differences = []
    similarities = []
    unique_to_struct1 = []
    unique_to_struct2 = []

    if is_namedtuple(node1) and is_namedtuple(node2):
        fields1 = set(node1._fields)
        fields2 = set(node2._fields)
        common_fields = fields1.intersection(fields2)

        for field in common_fields:
            new_path = f"{path}.{field}" if path else field
            diff, sim, unique1, unique2 = compare_leaves(
                getattr(node1, field), getattr(node2, field), new_path, atol=atol
            )
            differences.extend(diff)
            similarities.extend(sim)
            unique_to_struct1.extend(unique1)
            unique_to_struct2.extend(unique2)

        unique_fields1 = fields1 - fields2
        unique_fields2 = fields2 - fields1
        unique_to_struct1 += [f"{path}.{field}" for field in unique_fields1]
        unique_to_struct2 += [f"{path}.{field}" for field in unique_fields2]

    elif isinstance(node1, dict) and isinstance(node2, dict):
        keys1 = set(node1.keys())
        keys2 = set(node2.keys())
        common_keys = keys1.intersection(keys2)

        for key in common_keys:
            new_path = f"{path}.{key}" if path else key
            diff, sim, unique1, unique2 = compare_leaves(node1[key], node2[key], new_path, atol=atol)
            differences.extend(diff)
            similarities.extend(sim)
            unique_to_struct1.extend(unique1)
            unique_to_struct2.extend(unique2)

        unique_keys1 = keys1 - keys2
        unique_keys2 = keys2 - keys1
        unique_to_struct1 += [f"{path}.{key}" for key in unique_keys1]
        unique_to_struct2 += [f"{path}.{key}" for key in unique_keys2]

    else:
        array1 = get_array(node1)
        array2 = get_array(node2)
        if array1 is not None and array2 is not None:
            are_close, comparison_result = compare_new(array1, array2)
            if are_close:
                similarities.append(path)
            else:
                differences.append(path)
        else:
            if array1 is None and array2 is None:
                # Both are non-array; consider them similar if they are exactly the same
                if node1 == node2:
                    similarities.append(path)
                else:
                    differences.append(path)
            elif array1 is None:
                unique_to_struct2.append(path)
            else:
                unique_to_struct1.append(path)

    return differences, similarities, unique_to_struct1, unique_to_struct2


def _compare_structs(struct1, struct2, accept_failure=False, atol=1e-5, verbose=False):
    differences, similarities, unique_to_struct1, unique_to_struct2 = compare_leaves(struct1, struct2, atol=atol)

    if len(unique_to_struct1) > 0 or len(unique_to_struct2) > 0:
        if verbose:
            print("Unique fields in struct1:", unique_to_struct1)
            print("Unique fields in struct2:", unique_to_struct2)

    if len(differences) > 0:
        if verbose:
            print(f"Fields that are close within tol={atol}:", similarities)
            print(f"Fields that are different within tol={atol}:", differences)
        return False
    else:
        if verbose:
            print(f"Fields that are close within tol={atol}:", similarities)
        return True


def compare_structs(struct1, struct2, min_tol=-10, max_tol=-4, accept_failure=False):

    for tol in range(min_tol, max_tol + 1):
        close_enough = _compare_structs(struct1, struct2, atol=10**tol, verbose=False)
        if close_enough:
            # run again in verbose mode
            _compare_structs(struct1, struct2, atol=10**tol, verbose=True)
            print(f"Comparison passed with tol=1e-{tol}.")
            return

    if not close_enough:
        # run again in verbose mode
        _compare_structs(struct1, struct2, atol=10**tol, verbose=True)
        msg = f"Comparison failed at 1e-{max_tol}."
        if not accept_failure:
            raise ValueError(msg)
        else:
            print("WARNING:", msg, "Accepting failure.")
