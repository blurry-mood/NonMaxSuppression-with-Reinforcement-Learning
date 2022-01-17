import time
from copy import deepcopy
from torchmetrics.detection.map import MAP
import cv2 as cv
import numpy as np

from typing import Any, Dict, Tuple
from gym import Env
from gym.spaces import Discrete, Box
import torch
from retinaface.model import get_trained_model
from os.path import join, split, splitext
from glob import glob

from iou_utils import jaccard

_CURRENT_DIR = split(__file__)[0]


def parse_label_file(path):
    with open(path, 'r') as f:
        coordinates = f.read().split('\n')
        try:
            coordinates.remove('')
        except:
            pass
        for i in range(len(coordinates)):
            coordinates[i] = list(map(int, coordinates[i].split()))
        return torch.FloatTensor(coordinates)


def get_bboxes(model, img, conf_thres):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = (img - np.array([104, 117, 123])).transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0)
    boxes, scores = model(img)

    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]

    return torch.cat((boxes, scores.unsqueeze(1)), dim=1)


class ModelEnv(Env):

    def __init__(self, split: str, img_size: int, conf_thres) -> None:
        super().__init__()

        self.map = MAP(box_format='xyxy', iou_thresholds=[0.5],
                       compute_on_step=True)
        self.conf_thres = conf_thres
        self._model = get_trained_model(img_size)
        self.image_paths = glob(join(_CURRENT_DIR, '..', 'dataset',
                                     f'WIDER_{split}', 'images', '**', '*.jpg'), recursive=True)
        assert self.image_paths != [], "No image was found !"

        self.img_size = img_size

        self.action_space = Discrete(2)
        self.observation_space = None

        cv.namedWindow("Validation", cv.WINDOW_GUI_NORMAL)

    def reset(self) -> Any:
        repeat = True
        while repeat:
            img = np.random.choice(self.image_paths)

            self.labels = parse_label_file(splitext(img)[0] + '.txt')
            self.img = cv.resize(cv.imread(img), (self.img_size, self.img_size))

            with torch.no_grad():
                bboxes = get_bboxes(self._model, self.img, self.conf_thres)
                self.bboxes = bboxes
                bboxes = bboxes[:, :-1]

            if bboxes.shape[0] > 0:
                repeat = False

        self.state = torch.zeros(bboxes.shape[0], bboxes.shape[0] + 1)
        self.state[:, :-1] = jaccard(bboxes, bboxes)
        self.state[:, -1] = bboxes[:, -1]  # scores

        self.observation_space = Box(low=0, high=self.img_size, shape=self.state.shape)

        self.current_box = 0

        return deepcopy(self.state)

    def _compute_reward(self):
        mask = self.state.sum(dim=1) > 0
        bboxes = self.bboxes[mask]
        boxes = bboxes[:, :4]
        scores = bboxes[:, 4]
        preds = {'boxes': boxes, 'scores': scores, 'labels': torch.zeros(scores.shape[0]).long()}
        target = {'boxes': self.labels, 'labels': torch.zeros(len(self.labels)).long()}
        return self.map([preds], [target])['map_50'].item()

    def step(self, action):
        assert action == 0 or action == 1, 'Invalid action, choose either 0 or 1'

        if self.current_box == self.state.shape[0]:
            reward = self._compute_reward()
            return deepcopy(self.state), reward, True, {}

        if action == 0:
            self.state[self.current_box] = 0
        else:
            self.state[self.current_box] = -1

        self.current_box += 1

        return deepcopy(self.state), 0, False, {}

    def render(self, mode=''):

        img_cp = deepcopy(self.img)
        mask = self.state[:self.current_box].sum(dim=1) != 0
        bboxes = self.bboxes[:self.current_box]
        bboxes = bboxes[mask]
        boxes = bboxes[:, :4]
        for label in boxes:
            x1, y1, x2, y2 = map(int, label)
            cv.rectangle(img_cp, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
        cv.imshow('Validation', img_cp)

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            cv.destroyAllWindows()
            print("Quitting ...")
            exit(0)

    def close(self) -> None:
        cv.destroyAllWindows()
        return super().close()


# env = ModelEnv('train', 256, 0.5)
# state = env.reset()
# done = False
#
# while not done:
#     _, _, done, _ = env.step(env.action_space.sample())
#     env.render()
#     time.sleep(1)
