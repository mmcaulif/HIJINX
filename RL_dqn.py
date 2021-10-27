import os
import numpy as np

os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')  # tell JAX to use CPU, cpu is faster on small networks

import jax
import jax.nn
import jax.numpy as jnp
import haiku as hk
from collections import deque
import optax
import gym
import random

#hyperparameters
LEARNING_RATE = 3e-3
GAMMA = 0.95
BUFFER_SIZE = 100000
TRAIN_EPISODES = 1000000

#functions

"""def huber_loss(error):
    if error <= 1:
        huber_loss = jnp.square(error) * 0.5
    else:
        huber_loss = error - 0.5

    return huber_loss"""

@jax.grad
def loss(params,s_t,a_t,r_t,s_tp1):
    Q_s = forward(params, jnp.asarray(s_t))
    Q_sp1 = forward(params, jnp.asarray(s_tp1))
    td_e = jnp.max(Q_s) - (r_t + jnp.max(Q_sp1) * GAMMA)
    return td_e

@jax.jit    #sped it up maybe 20x fold
def update(params, optim_state, batch):
    grads = loss(params,batch[0],batch[1],batch[2],batch[3])
    updates, new_optim_state = optim.update(grads, optim_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_optim_state

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

#environment:
env = gym.make('CartPole-v1')
#experience replay:
replay_buffer = deque(maxlen=BUFFER_SIZE)
#neural network:
params, forward = net.init(jax.random.PRNGKey(42), jnp.ones(4)), jax.jit(hk.without_apply_rng(net).apply)
#optimiser:
optim = optax.adam(learning_rate=3e-3)
optim_state = optim.init(params)

s_t = env.reset()
r_total = 0

for i in range(1,TRAIN_EPISODES):
    while True:

        if len(replay_buffer) < BUFFER_SIZE: #basic explore system
            a_t = env.action_space.sample()
        else:
            a_t = int(jnp.max(forward(params, jnp.asarray(s_t))))

        s_tp1, r_t, done, _ = env.step(a_t)

        if done:
            break

        replay_buffer.append([s_t,a_t,r_t,s_tp1])
        batch = random.sample(replay_buffer, k=1)[0]
        update(params, optim_state, batch)

        s_t = s_tp1
        r_total = r_total + r_t

    if i % 100 == 0:
        print("Episode:", i, "is done, return =", r_total)

    r_total = 0
    s_t = env.reset()

env.close()
