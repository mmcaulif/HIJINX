import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')  # tell JAX to use CPU, cpu is faster on small networks

import numpy as np
import optax
import jax
import jax.nn
import jax.numpy as jnp
import haiku as hk
import torch
from torchvision import transforms, datasets

batch_size = 1  #currently program is only set up to handle one batch at a time, should aim to improve this to batch_size = ~8
EPOCHS = 3  #epoch  = ~60,000 trainsteps

train = datasets.MNIST("", train=True, download=True,transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True,transform = transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

@hk.transform
def forward(X):
    mlp = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(10),
    ])
    return mlp(X)

#haiku nn initialisation
key = jax.random.PRNGKey(seed=13)
input_size = jnp.ones([batch_size, 28 * 28])
params = forward.init(key, input_size)
forward = hk.without_apply_rng(forward).apply

@jax.grad
def mse(params, X, y):
    X = X.ravel()
    err = forward(params, X) - y
    return jnp.mean(jnp.square(err))    #mse

@jax.jit
def update(params, opt_state, grads):
    updates, opt_state = optimiser.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

#optimiser initialisation
optimiser = optax.sgd(learning_rate=0.001)
opt_state = optimiser.init(params)

count, correct = 0, 0

for epoch in range(EPOCHS):
    for data in trainset:
        X, y = jnp.array(data[0]), jnp.array(data[1])
        X = X.ravel()   #flatten X
        y = int(y)
        prediction = jnp.argmax(forward(params, X))

        if(count % 100 == 0 and count != 0):
            print("Trainstep:", count, " NN saids:", prediction, " Answer:", y)

        if(prediction != y):
            y_onehot = np.zeros(10)
            y_onehot[y] = 1
            y = jnp.array(y_onehot)
            grads = mse(params, X, y)
            params, opt_state = update(params, opt_state, grads)

        else:
            correct += 1

        count += 1

    success_rate = correct/count
    print("Epoch finished, success rate: ", success_rate)
    count, correct = 0, 0