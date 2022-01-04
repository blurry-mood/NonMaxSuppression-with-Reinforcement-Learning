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
        return coordinates

def get_n_boxes(model, imgsz):
    with torch.no_grad():
        test = torch.rand(1, 3, imgsz, imgsz)
        boxes, _ = model(test)
        return boxes.shape[0]

def get_bboxes(model, img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = (img - np.array([104, 117, 123])).transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0)
    boxes, scores = model(img)
    return np.concatenate((boxes.numpy(), scores.numpy()[:,None]), axis=1)


class ModelEnv(Env):

    def __init__(self, split: str, img_size: int) -> None:
        super().__init__()

        self.map = MAP(box_format='xyxy', iou_thresholds=[0.5],
                       compute_on_step=True)
        self._model = get_trained_model(img_size)
        self.image_paths = glob(join(_CURRENT_DIR, '..', 'dataset',
                                     f'WIDER_{split}', 'images', '**', '*.jpg'), recursive=True)
        assert self.image_paths != [], "No image was found !"

        self.n_boxes = get_n_boxes(self._model, img_size)
        self.img_size = img_size
        
        self.action_space = Discrete(2)
        """Each state is represented by the model output (boxes + scores) + one-hot vector (dropped, selected, current)"""
        self.observation_space = Box(
            low=0, high=img_size, shape=(self.n_boxes, 8), dtype=np.float32)
        
        cv.namedWindow("Validation", cv.WINDOW_GUI_NORMAL)

    def reset(self) -> Any:
        img = np.random.choice(self.image_paths)
        self.labels = parse_label_file(splitext(img)[0]+'.txt')
        self.img = cv.resize(cv.imread(img), (self.img_size, self.img_size))

        with torch.no_grad():
            bboxes = get_bboxes(self._model, self.img)

        self.current_box = 0
        self.bboxes = np.concatenate(
            (bboxes, np.zeros((bboxes.shape[0], 3))), axis=1)
        self._extract_state()

        return deepcopy(self.bboxes)

    def _compute_reward(self):
        mask = self.bboxes[:, -2] == 1
        bboxes = self.bboxes[mask]
        boxes = bboxes[:, :4]
        scores = bboxes[:, 4]
        preds = {'boxes': torch.from_numpy(
            boxes), 'scores': torch.from_numpy(scores), 'labels': torch.zeros(scores.shape[0]).long()}
        target = {'boxes': torch.FloatTensor(self.labels), 'labels': torch.zeros(len(self.labels)).long()}
        return self.map([preds], [target])['map_50'].item()

    def _extract_state(self):
        if self.current_box == self.bboxes.shape[0]:
            return True
        self.bboxes[self.current_box][-3:] = np.array([0, 0, 1])
        self.current_box += 1
        return False

    def step(self, action):
        assert action == 0 or action == 1, 'Invalid action, choose either 0 or 1'

        self.bboxes[self.current_box-1][-1] = 0
        self.bboxes[self.current_box-1][-2] = action
        self.bboxes[self.current_box-1][-3] = 1 - action

        done = self._extract_state()
        reward = self._compute_reward() if done else 0

        return deepcopy(self.bboxes), reward, done, {}

    def render(self):
        img_cp = deepcopy(self.img)
        mask = self.bboxes[:, -2] == 1
        bboxes = self.bboxes[mask]
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