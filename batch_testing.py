import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')  # tell JAX to use CPU, cpu is faster on small networks

import numpy as jnp
from collections import deque
import gym
import random

replay_buffer = deque(maxlen=8)
env = gym.make('CartPole-v0')
s_t = env.reset()

def batch_test(batch):
    s_tm1, a_tm1, r_tm1, s_t, done = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]
    print(s_tm1, s_t)
    product = s_t * s_tm1
    print(product)

for i in range(4):
        a_t = env.action_space.sample()

        s_tp1, r_t, done, info = env.step(a_t)

        replay_buffer.append([s_t, a_t, r_t, s_tp1, done])

        if len(replay_buffer) >= 2:
            batch = jnp.asarray(random.sample(replay_buffer, k=2), dtype=object)
            
        s_t = s_tp1

        if done:
            s_t = env.reset()

batch_test(batch)
