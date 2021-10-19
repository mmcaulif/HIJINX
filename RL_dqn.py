import os

import numpy as np

os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')  # tell JAX to use CPU, cpu is faster on small networks

import jax
import jax.nn
import jax.numpy as jnp
import haiku as hk
from collections import namedtuple
import optax
import coax
import gym
import sys

experience = jnp.zeros((0, 4))

print(jnp.shape(experience))
a = b = c = 1
d = jnp.array([[1,1]])
ones = jnp.array([[a,b,c,d]])
print(jnp.shape(ones))
experience = jnp.append(experience, ones, axis = 0)
print(experience, len(experience))

sys.exit()


def ReplayBuffer(exp, s, a, r, s_next):
    if len(exp) <= 100000:
        transition = jnp.array([[s, a, r, s_next]], dtype=int)
        exp = jnp.append(exp, transition, axis=0)
        return exp

    else:
        exp = jnp.delete(exp, 0)
        transition = jnp.array((s, a, r, s_next))
        exp = jnp.append(exp, transition, axis=0)
        return exp

env = gym.make('CartPole-v1')

@hk.transform
def func(S):
    seq = hk.Sequential([
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(env.action_space.n, w_init=jnp.zeros),
    ])
    return seq(S)

params = func.init(jax.random.PRNGKey(42), jnp.ones(4))

forward = jax.jit(hk.without_apply_rng(func).apply)

#q_table = forward(params, s_t)
#a_t = int(jnp.argmax(q_table))
#s_tp1, r_t, done, _ = env.step(a_t)

s_t = env.reset()
for i in range(100):
    a_t = env.action_space.sample() # your agent here (this takes random actions)
    s_tp1, r_t, done, _ = env.step(a_t)
    print(i)
    experience = ReplayBuffer(experience, s_t, a_t, r_t, s_tp1)

    s_t = s_tp1

    if done:
        s_t = env.reset()

env.close()

print(experience)
