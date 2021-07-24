import os
#os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU

import jax
import jax.numpy as jnp
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1)

param = {
    'w': jnp.array([0.]),   #no idea why but this needs to be an array, use jnp.zeros(X.shape[1:]) for features>1
    'b': 0.
}

@jax.jit
def model(params, X):
    return jnp.dot(X, params['w']) + params['b']

@jax.grad
def mse(params, X, y):
    diff = model(params, X) - y
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

print("Y =",float(param['w']), "* X +", float(param['b']))