import jax
import jax.numpy as jnp
import jax.nn
import numpy as np
import haiku as hk
import optax
import gym

#initialisations
env = gym.make('LunarLander-v2')
key = jax.random.PRNGKey(seed=7)
obs = jnp.ones(8)

#hyperparameters
TRAIN_STEPS = jnp.power(10, 6)
LEARNING_RATE = 0.003
GAMMA = 0.99

exploration = 1
exploration_rate = 0.99

@hk.transform
def actor_net(S):
    forward = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(16), jax.nn.relu,
        hk.Linear(4), jax.nn.sigmoid
    ])
    return forward(S)

a_params = actor_net.init(key, obs)
actor_forward = jax.jit(hk.without_apply_rng(actor_net).apply)

actor_optim = optax.adam(learning_rate=LEARNING_RATE)
a_opt_state = actor_optim.init(a_params)

@hk.transform
def critic_net(S):
    forward = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(16), jax.nn.relu,
        hk.Linear(1),
    ])
    return forward(S)

c_params = critic_net.init(key, obs)
critic_forward = jax.jit(hk.without_apply_rng(critic_net).apply)

critic_optim = optax.adam(learning_rate=LEARNING_RATE)
c_opt_state = actor_optim.init(c_params)

def advantage(r, done, next_s, s):
    adv = (r + (1- done) * GAMMA * critic_forward(c_params,next_s)) - critic_forward(c_params,s)
    c_loss = jnp.power(adv,2)
    a_probs = actor_forward(a_params, s)
    log_a_probs = jnp.log(a_probs)
    a_loss = -(log_a_probs * adv)
    ent = a_probs * log_a_probs
    return c_loss, a_loss, ent

def loss(c_loss, a_loss, ent):
    t_loss = a_loss + 0.5 * c_loss - 0.001 * ent
    return t_loss

next_obs = env.reset()

for i in range(TRAIN_STEPS):
    obs = next_obs
    exploration = exploration * exploration_rate
    if np.random.random_sample() > exploration:
        action = actor_forward(obs)
    else:
        action = env.action_space.sample()

    next_obs, reward, done, _ = env.step(action)

    #print(i, reward)

    if done:
        obs = env.reset()

    critic_loss, actor_loss, entropy = advantage(reward, done, next_obs, obs)
    total_loss = loss(critic_loss, actor_loss, entropy)

    print("Loss:", total_loss)

    """c_updates, c_opt_state = critic_optim.update(total_loss_grad, c_opt_state, c_params)
    c_params = optax.apply_updates(c_params, c_updates)

    a_updates, a_opt_state = critic_optim.update(total_loss_grad, a_opt_state, a_params)
    a_params = optax.apply_updates(a_params, a_updates)"""

env.close()

