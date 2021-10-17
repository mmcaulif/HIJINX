import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')  # tell JAX to use CPU, cpu is faster on small networks

import jax
import jax.nn
import jax.numpy as jnp
import haiku as hk
from collections import namedtuple
import optax
import coax
import gym

tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
replay = coax.experience_replay.SimpleReplayBuffer(capacity=100000)

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

obs = env.reset()

s_t = obs

params = func.init(jax.random.PRNGKey(42), jnp.ones(4))

forward = jax.jit(hk.without_apply_rng(func).apply)

q_table = forward(params, obs)

a_t = int(jnp.argmax(q_table))

s_tp1, r_t, d, _ = env.step(a_t)

tracer.add(s_t, a_t, r_t, d, s_tp1)

while tracer:
    replay.add(tracer.pop())

print(replay)

#print(replay.sample(batch_size=1))

for i in range(10000):
    while not d:
        obs = env.reset()
