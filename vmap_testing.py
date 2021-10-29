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

#environment:
env = gym.make('CartPole-v1')
#experience replay:
replay_buffer = deque(maxlen=5000)

s_t = env.reset()
for i in range(1,5):
    a_t = env.action_space.sample()
    s_tp1, r_t, done, _ = env.step(a_t)
    replay_buffer.append([s_t, a_t, r_t, s_tp1, done])
    print(s_t, a_t, r_t, s_tp1, done)
    s_t = s_tp1

    if done:
        s_t = env.reset()

env.close()

batch = random.sample(replay_buffer, k=2)

print(jnp.shape(batch))

v_add_data = jax.vmap(add_data, in_axes=None, out_axes=0)

print(v_add_data(batch[0],batch[1],batch[2],batch[3]))

