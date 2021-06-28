import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')  # tell JAX to use CPU
# os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import jax
import jax.numpy as jnp
import coax
import haiku as hk
from numpy import prod
import optax

#echo $DISPLAY
#export DISPLAY=192.168.1.19:0.0

# pick environment
env = gym.make('CartPole-v1')
env = coax.wrappers.TrainMonitor(env)


def func_v(S, is_training):
    # custom haiku function
    value = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    return value(S)  # output shape: (batch_size,)


def func_pi(S, is_training):
    logits = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(env.action_space.n, w_init=jnp.zeros)
    ))
    return {'logits': logits(S)}


# function approximators
v = coax.V(func_v, env)
pi = coax.Policy(func_pi, env)


# specify how to update policy and value function
vanilla_pg = coax.policy_objectives.VanillaPG(pi, optimizer=optax.adam(0.001))
simple_td = coax.td_learning.SimpleTD(v, optimizer=optax.adam(0.002))


# specify how to trace the transitions
tracer = coax.reward_tracing.NStep(n=5, gamma=0.95)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=256)


for ep in range(250):
    state = env.reset()
    for t in range(env.spec.max_episode_steps):
        action, logp = pi(state, return_logp=True)
        state_next, reward, done, info = env.step(action)

        # add transition to buffer
        # N.B. vanilla-pg doesn't use logp but we include it to make it easy to
        # swap in another policy updater that does require it, e.g. ppo-clip
        tracer.add(state, action, reward, done, logp)
        while tracer:
            buffer.add(tracer.pop())

        # update
        if len(buffer) == buffer.capacity:
            for _ in range(4 * buffer.capacity // 32):  # ~4 passes
                transition_batch = buffer.sample(batch_size=32)
                metrics_v, td_error = simple_td.update(transition_batch, return_td_error=True)
                metrics_pi = vanilla_pg.update(transition_batch, td_error)
                env.record_metrics(metrics_v)
                env.record_metrics(metrics_pi)

            buffer.clear()

        if done:
            break

        state = state_next

#eval environment
env.reset()
print("Evaluating")
for ep in range (30):
    for t in range(env.spec.max_episode_steps):
        action = pi.mode(state)
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            env.reset()
            break