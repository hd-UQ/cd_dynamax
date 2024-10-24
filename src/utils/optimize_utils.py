import jax.numpy as jnp
import jax.random as jr
import optax
from jax import lax, value_and_grad, tree_map

# From dynamax
from dynamax.utils.utils import pytree_len
# Import original dynamax run_sgd, and sample_minibatches
from dynamax.utils.optimize import run_sgd as dynamax_run_sgd
from dynamax.utils.optimize import sample_minibatches


def make_optimizer(
    initial_learning_rate=1e-1,
    decay_factor=0.5,
    epochs_per_step=1500,
    num_epochs=5000,
    use_lr_scheduler=True,
    clip_norm=1.0,
):
    if use_lr_scheduler:
        # Define the boundaries where the decay should happen
        boundaries = [epochs_per_step * i for i in range(1, num_epochs // epochs_per_step + 1)]

        # Create a step decay learning rate schedule
        scheduler = optax.piecewise_constant_schedule(
            init_value=initial_learning_rate, boundaries_and_scales={boundary: decay_factor for boundary in boundaries}
        )
    else:
        # Use a constant learning rate if scheduler is not used
        scheduler = optax.constant_schedule(initial_learning_rate)

    # Define an optimizer with optional gradient clipping and learning rate schedule
    optimizer_steps = []
    if clip_norm is not None:
        optimizer_steps.append(optax.clip_by_global_norm(clip_norm))
    optimizer_steps.append(optax.scale_by_adam())
    optimizer_steps.append(optax.scale_by_schedule(scheduler))
    optimizer_steps.append(optax.scale(-1.0))

    my_optimizer = optax.chain(*optimizer_steps)

    return my_optimizer


# cd-dynamax wrapper,
# to be able to return the history of parameters and gradients
def run_sgd(loss_fn,
            params,
            dataset,
            optimizer=optax.adam(1e-3),
            batch_size=1,
            num_epochs=50,
            shuffle=False,
            return_param_history=False,
            return_grad_history=False,
            key=jr.PRNGKey(0)):
    """
    Note that batch_emissions is initially of shape (N,T)
    where N is the number of independent sequences and
    T is the length of a sequence. Then, a random susbet with shape (B, T)
    of entire sequence, not time steps, is sampled at each step where B is
    batch size.

    Args:
        loss_fn: Objective function.
        params: initial value of parameters to be estimated.
        dataset: PyTree of data arrays with leading batch dimension
        optmizer: Optimizer.
        batch_size: Number of sequences used at each update step.
        num_iters: Iterations made on only one mini-batch.
        shuffle: Indicates whether to shuffle emissions.
        return_param_history: Indicates whether to return history of parameters.
        return_grad_history: Indicates whether to return history of gradients.
        key: RNG key.

    Returns:
        hmm: HMM with optimized parameters.
        losses: Output of loss_fn stored at each step.
    """

    # If both return_param_history and return_grad_history are False,
    # call original run_sgd function
    if not return_param_history and not return_grad_history:
        params, losses = dynamax_run_sgd(
            loss_fn,
            params,
            dataset,
            optimizer,
            batch_size,
            num_epochs,
            shuffle,
            key
        )
    
    # Replica of dynamax_run_sgd function with modifications
    else:
        opt_state = optimizer.init(params)
        num_batches = pytree_len(dataset)
        num_complete_batches, leftover = jnp.divmod(num_batches, batch_size)
        num_batches = num_complete_batches + jnp.where(leftover == 0, 0, 1)
        loss_grad_fn = value_and_grad(loss_fn)

        # implement this for ReduceLRonPlateau
        # https://optax.readthedocs.io/en/latest/_collections/examples/contrib/reduce_on_plateau.html
        if batch_size >= num_batches:
            shuffle = False

        def train_step(carry, key):
            params, opt_state = carry
            sample_generator = sample_minibatches(key, dataset, batch_size, shuffle)

            def cond_fun(state):
                itr, params, opt_state, avg_loss, grads = state
                return itr < num_batches

            def body_fun(state):
                itr, params, opt_state, avg_loss, grads = state
                minibatch = next(sample_generator)  ## TODO: Does this work inside while_loop??
                this_loss, grads = loss_grad_fn(params, minibatch)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return itr + 1, params, opt_state, (avg_loss * itr + this_loss) / (itr + 1), grads

            init_val = (0, params, opt_state, 0.0, params) # last param is dummy for grads.
            _, params, opt_state, avg_loss, grads = lax.while_loop(cond_fun, body_fun, init_val)

            return (params, opt_state), (avg_loss, params, grads)

        keys = jr.split(key, num_epochs)
        (params, _), (losses, param_history, grad_history) = lax.scan(train_step, (params, opt_state), keys)
        
    # If not interested in history of parameters
    if not return_param_history:
        param_history = None
    # If not interested in history of gradients
    if not return_grad_history:
        grad_history = None

    return params, losses, param_history, grad_history
