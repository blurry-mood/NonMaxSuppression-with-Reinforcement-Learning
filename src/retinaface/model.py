""" This repository was taken from https://github.com/biubug6/Pytorch_Retinaface"""

import os
from .models.retina import Retina
from .config import cfg_mnet
import torch

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    # unused_pretrained_keys = ckpt_keys - model_keys
    # missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def _load_model(model, pretrained_path,):
    print('Loading pretrained model from {}'.format(pretrained_path))

    pretrained_dict = torch.load(
        pretrained_path, map_location=lambda storage, loc: storage)

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def get_trained_model(width, height=None):
    if height is None:
        height = width
    PATH = os.path.split(os.path.abspath(__file__))[0]

    cfg = cfg_mnet

    # net and model
    # Update input image width & height as you wish
    net = Retina(cfg=cfg, phase='test', width=width, height=height)
    net = _load_model(net, os.path.join(PATH, 'weights', 'mobilenet0.25_Final.pth'))
    net.eval()

    return net