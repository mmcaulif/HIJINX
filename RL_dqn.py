import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
from jax.lax import stop_gradient
import haiku as hk
from collections import deque
from typing import NamedTuple
import optax
import gym
from gym.wrappers import RecordEpisodeStatistics
import random
import os

os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')  # tell JAX to use CPU, cpu is faster on small networks

# Hyper parameters from stable baselines3 - https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
GAMMA = 0.99
BUFFER_SIZE = 1000000
TRAIN_STEPS = 300000
TARGET_UPDATE = 10000
VERBOSE_UPDATE = 1000
EPSILON = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
TRAIN_START = 50000
LR_SCHEDULE = optax.linear_schedule(LEARNING_RATE, 0, TRAIN_STEPS, TRAIN_START)

# Functions
class Transition(NamedTuple):
	s: list # state
	a: int # action
	r: float # reward
	s_p: list # next state
	d: int # done

@jax.vmap
def q_loss_fn(Q_s, Q_sp1, s_t, a_t, r_t, s_tp1, done):
    Q_target = r_t + GAMMA * Q_sp1.max() * (1 - done)
    return (Q_s[a_t] - Q_target)

@jax.jit
def mse_loss(params, target_params, s_t, a_t, r_t, s_tp1, done):
    Q_s = forward(params, s_t)
    Q_sp1 = stop_gradient(forward(target_params, s_tp1))
    losses = q_loss_fn(Q_s, Q_sp1, s_t, a_t, r_t, s_tp1, done)
    return 0.5 * jnp.square(losses).mean()

@jax.jit  # sped it up maybe 20x fold
def update(params, target_params, optim_state, batch):
    s_t = jnp.array(batch.s, dtype = jnp.float32)
    a_t = jnp.array(batch.a, dtype = jnp.int32)
    r_t = jnp.array(batch.r, dtype = jnp.float32)
    s_tp1 = jnp.array(batch.s_p, dtype = jnp.float32)
    done = jnp.array(batch.d, dtype = jnp.float32)

    loss, grads = jax.value_and_grad(mse_loss)(params, target_params, s_t, a_t, r_t, s_tp1, done)   # find jax equivelant to this:
    updates, optim_state = optimizer.update(grads, optim_state, params) # https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
    params = optax.apply_updates(params, updates)
    return params, optim_state


def epsilon_greedy(epsilon):    #need to make anneal from a max to a min instead of current method
    rand = random.random()
    if rand < epsilon:
        a_t = env.action_space.sample()
    else:
        a_t = int(jnp.argmax(forward(params, jnp.asarray(s_t))))
    epsilon = epsilon * 0.99
    return a_t, epsilon


# initialisations
@hk.transform   #stable baselines3 dqn network is input_dim, 64, 64, output_dim
def net(S):
    seq = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(env.action_space.n),  # , w_init=jnp.zeros
    ])
    return seq(S)


# environment:
env = gym.make('CartPole-v0')
env = RecordEpisodeStatistics(env)
# experience replay:
replay_buffer = deque(maxlen=BUFFER_SIZE)
# neural network:
params, forward = net.init(jax.random.PRNGKey(42), jnp.ones(4)), hk.without_apply_rng(net).apply
target_params = hk.data_structures.to_immutable_dict(params)
# optimiser:
optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(learning_rate=LR_SCHEDULE))
optim_state = optimizer.init(params)

s_t = env.reset()
G = []

for i in range(1, TRAIN_STEPS):
    a_t, EPSILON = epsilon_greedy(EPSILON)
    s_tp1, r_t, done, info = env.step(a_t)

    replay_buffer.append([s_t, a_t, r_t, s_tp1, done])

    if i > TRAIN_START and len(replay_buffer) > BATCH_SIZE:
        batch = Transition(*zip(*random.sample(replay_buffer, k=BATCH_SIZE)))
        params, optim_state = update(params, target_params, optim_state, batch)

    s_t = s_tp1

    if i % TARGET_UPDATE == 0:
        target_params = hk.data_structures.to_immutable_dict(params)

    if done:
        G.append(int(info['episode']['r']))
        s_t = env.reset()

    if i % VERBOSE_UPDATE == 0:
        avg_G = sum(G[-10:])/10
        print("Timestep: {}, Average return: {}".format(i, avg_G))

env.close()
