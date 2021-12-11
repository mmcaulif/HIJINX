import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')  # tell JAX to use CPU, cpu is faster on small networks
import jax
import jax.numpy as jnp
import numpy as np
from collections import deque
from typing import NamedTuple
import haiku as hk
import gym
import random

replay_buffer = deque(maxlen=8)
env = gym.make('CartPole-v0')
s_t = env.reset()

@jax.jit
def batch_test(s_tm1, s_t):
    #s_tm1, a_tm1, r_tm1, s_t, done = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]
    return jnp.mean(s_tm1 + s_t)

for i in range(4):
        a_t = env.action_space.sample()

        s_tp1, r_t, done, info = env.step(a_t)

        replay_buffer.append([s_t, a_t, r_t, s_tp1, done])           
            
        s_t = s_tp1

        if done:
            s_t = env.reset()

class Transition(NamedTuple):
	s_tm1: list # state
	a: int # action
	r: float # reward
	s_t: list # next state
	d: bool # done

batch = Transition(*zip(*random.sample(replay_buffer, k=4)))

s_tm1 = jnp.array(batch.s_tm1, dtype=jnp.float32)
a_t = jnp.array(batch.a, dtype = jnp.int32)
r_t = jnp.array(batch.r, dtype = jnp.float32)
s_t = jnp.array(batch.s_t, dtype=jnp.float32)
done = jnp.array(batch.d, dtype = jnp.float32)

@hk.transform   #stable baselines3 dqn network is input_dim, 64, 64, output_dim
def net(S):
    seq = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(env.action_space.n),  # , w_init=jnp.zeros
    ])
    return seq(S)
    
params, forward = net.init(jax.random.PRNGKey(42), jnp.ones(4)), hk.without_apply_rng(net).apply

Q_s = forward(params, s_t)

"""print("Action: ", a_t)
print("\nState:\n", s_t, "\nQ-Value:\n", Q_s)
print("\nPairs: ")
Q_s = jnp.array([Q_s[i][a_t[i]] for i in range(len(a_t))], dtype = jnp.float32)

print(Q_s)"""

@jax.vmap
def test_func(stm1, at, rt, st, d):
    qtm1, qt = forward(params, stm1), forward(params, st)
    return qtm1[at], qt[at]

Q_sm1, Q_s = test_func(s_tm1, a_t, r_t, s_t, done)

print(Q_sm1, Q_s)