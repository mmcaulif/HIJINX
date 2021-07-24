import os
#os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU

import jax
import jax.numpy as jnp
import haiku as hk
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1)

def forward(X):
    forward_pass = hk.Linear(1)
    return forward_pass(X).ravel()

key = jax.random.PRNGKey(seed=13)
param = hk.transform(forward).init(key, X)

forward = hk.without_apply_rng(hk.transform(forward)).apply

@jax.grad
def mse(params, X, y):
    diff = forward(params, X) - y
    l = jnp.mean(jnp.square(diff))
    print("Loss:", int(l))
    return l

#grad_fn = jax.grad(mse)

@jax.jit
def update(params, grads):
    return jax.tree_multimap(lambda p, g: p - 0.05 * g, params, grads)

for i in range(50):
    """loss = mse(param, X, y)
    print("Loss: ", loss)

    grad = grad_fn(param, X, y)"""
    grad = mse(param, X, y)
    param = update(param, grad)

print(param.values())