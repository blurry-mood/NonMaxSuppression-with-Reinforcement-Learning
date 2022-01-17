import time

from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from rl_algorithms import DQN
from torch import nn
import torch
from environment import ModelEnv
from config import *
from os.path import split, join

_CURRENT_DIR = split(__file__)[0]


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(nn.Conv2d(1, 32, 3, padding=5, dilation=3),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 64, 5, padding=5, dilation=3),
                                   nn.ReLU(),
                                   nn.AdaptiveAvgPool2d(1),
                                   nn.Flatten(),
                                   nn.ReLU(),
                                   nn.Linear(64, 2)
                                   )

    def forward(self, x):
        return self.model(x)


class MyAgent(DQN):

    def decode_state(self, state: torch.Tensor):
        if state is None:
            return None
        s = state.unsqueeze(0).unsqueeze(0).float()
        return s


env = ModelEnv('train', IMG_SIZE, CONF_THRES)
dqn = MyAgent(Model(), actions=list(range(env.action_space.n)), alpha=ALPHA, gamma=GAMMA, eps=EPS)
dqn.load(join(_CURRENT_DIR, '..', 'artifacts', 'dqn'))

losses = []
rewards = []

for i in range(EPISODES):
    state = env.reset()
    env.render()
    n = 0
    done = False
    _loss = 0
    with tqdm(desc=f'Episode {i + 1}') as pbar:
        while not done:
            n += 1
            action = dqn.take_action(state)
            state, reward, done, info = env.step(action)

            loss = dqn.update(state, reward)
            if loss is not None:
                _loss += loss
            env.render()

            pbar.update(1)
            pbar.set_postfix({'reward': reward, 'loss': loss, 'Number of bboxes': env.observation_space.shape[0]})

    dqn.buffer.buffer.clear()

    rewards.append(reward)
    losses.append(_loss)
    dqn.save(join(_CURRENT_DIR, '..', 'artifacts', 'dqn'))

env.close()

plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Reward per episode")
plt.savefig(join(_CURRENT_DIR, '..', 'artifacts', 'reward.png'))
plt.show()

plt.plot(losses)
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.title("Loss per episode")
plt.savefig(join(_CURRENT_DIR, '..', 'artifacts', 'loss.png'))
plt.show()