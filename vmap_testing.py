import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')  # tell JAX to use CPU, cpu is faster on small networks

import jax
import jax.nn
import jax.numpy as jnp
from collections import deque
import gym
import random

def add_data(a, b, c, d):
    sum = jnp.sum(a) + b + c + jnp.sum(d)
    return sum

replay_buffer = jnp.ones([10,5])

print(replay_buffer)
"""
batch = random.sample(replay_buffer, k=2)

print(jnp.shape(batch))

v_add_data = jax.vmap(add_data, in_axes=None, out_axes=0)

print(v_add_data(batch[0],batch[1],batch[2],batch[3]))
"""


