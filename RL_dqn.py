import jax
import jax.nn
import jax.numpy as jnp
from jax.lax import stop_gradient
import haiku as hk
from collections import deque
import optax
import gym
from gym.wrappers import RecordEpisodeStatistics
import random
import os

os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')  # tell JAX to use CPU, cpu is faster on small networks

# hyper parameters
LEARNING_RATE = 1e-3
GAMMA = 0.99
BUFFER_SIZE = 100000
TRAIN_EPS = 3000
TARGET_UPDATE = 10
VERBOSE_UPDATE = 25
EPSILON = 1
BATCH_SIZE = 8

# functions
mse_loss = lambda x, xp: (jnp.power((x - xp), 2)).sum().mean()  #.sum(-1)

@jax.grad
def loss(params, target_params, batch):
    s_t = batch[:,0]
    a_t = batch[:,1]
    r_t = batch[:,2]
    s_tp1 = batch[:,3]
    done = batch[:,4]
    Q_s = forward(params, jnp.asarray(s_t))
    Q_sp1 = forward(target_params, jnp.asarray(s_tp1))  # don't compute grads of target
    Q_target = r_t + jnp.max(Q_sp1) * GAMMA * (1 - done)
    td_mse = mse_loss(Q_s[a_t], stop_gradient(Q_target))
    return 0.5 * td_mse


@jax.jit  # sped it up maybe 20x fold
def update(params, target_params, optim_state, batch):
    grads = loss(params, target_params, batch)
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


# initialisations

@hk.transform
def net(S):
    seq = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(128), jax.nn.relu,
        hk.Linear(env.action_space.n),  # , w_init=jnp.zeros
    ])
    return seq(S)


# environment:
env = gym.make('CartPole-v1')
env = RecordEpisodeStatistics(env)
# experience replay:
replay_buffer = deque(maxlen=BUFFER_SIZE)
# neural network:
params, forward = net.init(jax.random.PRNGKey(42), jnp.ones(4)), jax.jit(hk.without_apply_rng(net).apply)
target_params = params
# optimiser:
optimizer = optax.rmsprop(learning_rate=LEARNING_RATE)
optim_state = optimizer.init(params)

s_t = env.reset()

avg_eps_return, return_sum = 0, 0

for i in range(1, TRAIN_EPS):
    done = False
    while not done:
        a_t = epsilon_greedy(EPSILON)
        s_tp1, r_t, done, info = env.step(a_t)

        replay_buffer.append([s_t, a_t, r_t, s_tp1, done])
        if len(replay_buffer) > BATCH_SIZE:
            batch = random.sample(replay_buffer, k=BATCH_SIZE)
            print(batch[:, 0])
            params, optim_state = update(params, target_params, optim_state, batch)

        s_t = s_tp1
        EPSILON = EPSILON * 0.99

        if i % TARGET_UPDATE == 0:
            target_params = params

        if done:
            return_sum += int(info['episode']['r'])
            if i % VERBOSE_UPDATE == 0:
                avg_eps_return = return_sum / VERBOSE_UPDATE
                print("Episode: {}, Average return: {}".format(i, avg_eps_return))
                return_sum = 0

    s_t = env.reset()

env.close()
