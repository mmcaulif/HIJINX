import jax
import jax.numpy as jnp
import jax.nn
import numpy as np
import haiku as hk
import optax
import gym

#initialisations
env = gym.make('CartPole-v1')
key = jax.random.PRNGKey(seed=7)

#hyperparameters
TRAIN_STEPS = jnp.power(10, 6)
LEARNING_RATE = 0.003
GAMMA = 0.99

exploration = 1
exploration_rate = 0.99

@hk.transform
def a2c_net(S):
    actor_forward = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(16), jax.nn.relu,
        hk.Linear(env.action_space.n), jax.nn.sigmoid
    ])
    critic_forward = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(16), jax.nn.relu,
        hk.Linear(1)
    ])
    return actor_forward(S), critic_forward(S)

params = a2c_net.init(key, obs)
forward = jax.jit(hk.without_apply_rng(a2c_net).apply)

optim = optax.adam(learning_rate=LEARNING_RATE)
opt_state = optim.init(params)

def critic_loss(r, params, s):
    _, v_t = forward(params, s)
    adv = r - v_t
    return 0.5 * jnp.pow(adv, 2)

def loss(c_loss, a_loss, ent):
    t_loss = a_loss + 0.5 * c_loss - 0.001 * ent
    return t_loss

next_obs = env.reset()

for i in range(TRAIN_STEPS):
    obs = next_obs
    exploration = exploration * exploration_rate
    if np.random.random_sample() > exploration:
        act_probs, _ = forward(params, obs)
    else:
        action = env.action_space.sample()

    next_obs, reward, done, _ = env.step(action)

    #print(i, reward)

    if done:
        obs = env.reset()


    print("Loss:", total_loss)

    updates, opt_state = optim.update(total_loss_grad, opt_state, params)
    params = optax.apply_updates(params, updates)

env.close()

