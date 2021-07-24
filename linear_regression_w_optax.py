import os
#os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU

import jax
import jax.numpy as jnp
import haiku as hk
import optax
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1)

def forward(X):
    forward_pass = hk.Linear(1)
    return forward_pass(X).ravel()

key = jax.random.PRNGKey(seed=13)
param = hk.transform(forward).init(key, X)
forward = hk.without_apply_rng(hk.transform(forward)).apply

tx = optax.sgd(learning_rate=0.05)
opt_state = tx.init(param)

@jax.grad
def mse(params, X, y):
    diff = forward(params, X) - y
    l = jnp.mean(jnp.square(diff))
    print("Loss:", int(l))
    return l

#grad_fn = jax.grad(mse)

@jax.jit
def update(params, opt_state, grads):
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

for i in range(50):
    """loss = mse(param, X, y)
    print("Loss: ", loss)

    grad = grad_fn(param, X, y)"""
    grad = mse(param, X, y)
    param, opt_state = update(param, opt_state, grad)

print(param.values())