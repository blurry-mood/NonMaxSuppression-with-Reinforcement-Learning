from tqdm.auto import tqdm

from rl_algorithms import DQN
from torch import nn
import numpy as np
import torch
from environment import ModelEnv

from os.path import split, join

_CURRENT_DIR = split(__file__)[0]

ALPHA = 2e-1
GAMMA = 0.99
EPS = 1e-1
ITERS = 2


class Model(nn.Module):

    def __init__(self, in_dims, n_actions):
        super().__init__()

        self.model = nn.Sequential(nn.Linear(in_dims, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, n_actions)
                                   )

    def forward(self, x):
        return self.model(x)


class MyAgent(DQN):

    def decode_state(self, bboxes: np.ndarray):
        mask = bboxes[:, 5] == 0
        # bboxes[mask] = 0
        s = np.delete(bboxes, 5, axis=1)
        s = torch.from_numpy(s).flatten().unsqueeze(0).float()
        return s


img_size = 128

env = ModelEnv('train', img_size)
dqn = MyAgent(Model(env.n_boxes * 7, 2), actions=list(range(env.action_space.n)), alpha=ALPHA, gamma=GAMMA, eps=EPS)
dqn.load(join(_CURRENT_DIR, '..', 'artifacts', 'dqn'))

for i in range(ITERS):
    state = env.reset()
    env.render()
    n = 0
    done = False
    reds = 0
    ones = 0
    with tqdm(desc=f'Episode {i}') as pbar:
        while not done:
            n += 1
            action = dqn.take_action(state)
            ones += action
            state, reward, done, info = env.step(action)
            reward += 1 / (ones + 1)
            dqn.update(state, reward)
            env.render()

            reds += reward

            pbar.update(1)
            pbar.set_postfix({'reward': reds, 'ones': ones / n})

    dqn.save(join(_CURRENT_DIR, '..', 'artifacts', 'dqn'))

env.close()
