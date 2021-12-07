import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')  # tell JAX to use CPU, cpu is faster on small networks
import jax
import jax.numpy as jnp
import numpy as np
from collections import deque
from typing import NamedTuple
import gym
import random

replay_buffer = deque(maxlen=8)
env = gym.make('CartPole-v0')
s_t = env.reset()

@jax.jit
def batch_test(s_tm1, s_t):
    #s_tm1, a_tm1, r_tm1, s_t, done = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]
    return jnp.mean(s_tm1 + s_t)

for i in range(4):
        a_t = env.action_space.sample()

        s_tp1, r_t, done, info = env.step(a_t)

        replay_buffer.append([s_t, a_t, r_t, s_tp1, done])           
            
        s_t = s_tp1

        if done:
            s_t = env.reset()

class Transition(NamedTuple):
	s_tm1: list # state
	a: int # action
	r: float # reward
	s_t: list # next state
	d: bool # done

batch = Transition(*zip(*random.sample(replay_buffer, k=4)))

print("B: ", batch)

s_tm1 = jnp.array(batch.s_tm1, dtype=jnp.float32)
s_t = jnp.array(batch.s_t, dtype=jnp.float32)

print("S1: ", s_tm1)
print("S2: ", s_t)

print(batch_test(s_tm1, s_t))
