import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')     # tell JAX to use CPU
#os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import coax
import gym
import haiku as hk
import jax
import jax.numpy as jnp
from coax.value_losses import mse
from optax import adam

#echo $DISPLAY
#export DISPLAY=192.168.1.18:0.0

# the name of this script
name = 'dqn'

# the cart-pole MDP
#env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')
env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=None, tensorboard_write_all=False)

def func(S, is_training):
    """ type-2 q-function: s -> q(s,.) """
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(env.action_space.n, w_init=jnp.zeros) #initialises weights as 0, otherwise initialises them randomly
    ))
    return seq(S)


# value function and its derived policy
q = coax.Q(func, env)
pi = coax.BoltzmannPolicy(q, temperature=0.1)

# target network
q_targ = q.copy()

# experience tracer
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=100000)

# updater
qlearning = coax.td_learning.QLearning(q, q_targ=q_targ, loss_function=mse, optimizer=adam(0.001))

TRAIN_EPISODES = 100

# train
for ep in range(TRAIN_EPISODES):
    s = env.reset()
    # pi.epsilon = max(0.01, pi.epsilon * 0.95)
    # env.record_metrics({'EpsilonGreedy/epsilon': pi.epsilon})

    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        s_next, r, done, info = env.step(a)

        # extend last reward as asymptotic best-case return
        if t == env.spec.max_episode_steps - 1:
            assert done
            r = 1 / (1 - tracer.gamma)  # gamma + gamma^2 + gamma^3 + ... = 1 / (1 - gamma)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, done)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= 100:
            transition_batch = buffer.sample(batch_size=32)
            metrics = qlearning.update(transition_batch)
            env.record_metrics(metrics)

        # sync target network
        q_targ.soft_update(q, tau=0.01)

        if done:
            break

        s = s_next

    # early stopping
    if env.avg_G > env.spec.reward_threshold:
        print("Early stop")
        break

env.reset()

for ep in range (10):

    for t in range(env.spec.max_episode_steps):
        a = pi.mode(s)
        s, r, done, info = env.step(a)

        env.render()

        if done:
            env.reset()
            break