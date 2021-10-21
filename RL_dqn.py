import os
import numpy as np

os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')  # tell JAX to use CPU, cpu is faster on small networks

import jax
import jax.nn
import jax.numpy as jnp
import haiku as hk
from collections import namedtuple, deque
import optax
import gym
import sys

#hyperparameters
"""LEARNING_RATE =
GAMMA =
BUFFER_SIZE =
TRAIN_STEPS = """

#functions
Transition = namedtuple('Transition',('state', 'action', 'reward', 'next_state'))

class ExperienceReplay(object):
    def __init__ (self, buffer_size):
        self.memory = deque([], maxlen=buffer_size)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#def loss():    #look into bellman equation before implementing

#initialisations

@hk.transform
def net(S):
    seq = hk.Sequential([
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(env.action_space.n, w_init=jnp.zeros),
    ])
    return seq(S)

env = gym.make('CartPole-v1')
replay_buffer = ExperienceReplay(buffer_size=100000)

params, forward = net.init(jax.random.PRNGKey(42), jnp.ones(4)), jax.jit(hk.without_apply_rng(net).apply)

optim = optax.adam(learning_rate=3e-3)
optim_state = optim.init(params)

#q_table = forward(params, s_t)
#a_t = int(jnp.argmax(q_table))
#s_tp1, r_t, done, _ = env.step(a_t)

s_t = env.reset()
for i in range(100):
    a_t = env.action_space.sample() # your agent here (this takes random actions)
    s_tp1, r_t, done, _ = env.step(a_t)
    print(i)

    replay_buffer.push(s_t, a_t, r_t, s_tp1)

    s_t = s_tp1

    if done:
        s_t = env.reset()

env.close()

print(replay_buffer)
