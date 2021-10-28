import os
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
TRAIN_STEPS = 1000000

eps = 1

#functions

"""def huber_loss(error):
    if error <= 1:
        huber_loss = jnp.square(error) * 0.5
    else:
        huber_loss = error - 0.5

    return huber_loss"""

@jax.grad
def loss(params,s_t,a_t,r_t,s_tp1,done):
    Q_s = forward(params, jnp.asarray(s_t))
    Q_sp1 = forward(params, jnp.asarray(s_tp1))  #don't compute grads of target
    Q_target = r_t + jnp.max(Q_sp1) * GAMMA * (1 - done)
    td_e = Q_s[a_t] - jax.lax.stop_gradient(Q_target)
    return td_e ** 2

@jax.jit    #sped it up maybe 20x fold
def update(params, optim_state, batch):
    grads = loss(params,batch[0],batch[1],batch[2],batch[3],batch[4])
    updates, optim_state = optimizer.update(grads, optim_state, params)
    params = optax.apply_updates(params, updates)
    return params, optim_state

def epsilon_greedy(eps):
    r = random.random()
    if r < eps:  # basic explore system
        a_t = env.action_space.sample()
    else:
        a_t = int(jnp.argmax(forward(params, jnp.asarray(s_t))))

    return a_t

#initialisations

@hk.transform
def net(S):
    seq = hk.Sequential([
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(env.action_space.n),  #, w_init=jnp.zeros
    ])
    return seq(S)

#environment:
env = gym.make('CartPole-v1')
#experience replay:
replay_buffer = deque(maxlen=BUFFER_SIZE)
#neural network:
params, forward = net.init(jax.random.PRNGKey(42), jnp.ones(4)), jax.jit(hk.without_apply_rng(net).apply)
#optimiser:
optimizer = optax.adam(learning_rate=3e-3)
optim_state = optimizer.init(params)

s_t = env.reset()
r_avg = 0
count = 0

for i in range(1,TRAIN_STEPS):
    """if count % 250 == 0:
        env.render()"""

    a_t = epsilon_greedy(eps)
    s_tp1, r_t, done, _ = env.step(a_t)

    replay_buffer.append([s_t, a_t, r_t, s_tp1, done])
    batch = random.sample(replay_buffer, k=1)[0]
    params, optim_state = update(params, optim_state, batch)

    s_t = s_tp1
    r_avg = r_avg + r_t

    if done:
        count = count + 1
        eps = eps * 0.99
        if count % 250 == 0:
            print("Episode:", count, ", Average Return:", r_avg/250)
            r_avg = 0

        s_t = env.reset()

env.close()
