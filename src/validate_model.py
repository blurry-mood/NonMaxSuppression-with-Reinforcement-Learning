from argparse import ArgumentParser
import numpy as np
from glob import glob
from os.path import join, split, splitext
import cv2 as cv

import torch
from torchvision.ops import batched_nms


from tqdm.auto import tqdm
from retinaface.model import get_trained_model


_CURRENT_DIR = split(__file__)[0]

parser = ArgumentParser(
    description="This script visualizes the WIDER Face dataset samples one at a time. To view the next image type `n` (next), and to quit type `q` (quit).")
parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                    help='Which dataset split to visualize: `train`, `val`, and `test`')
parser.add_argument('--imgsz', type=int, default='1024', help='Specify the image size (width=height)')
parser.add_argument('--conf-thres', type=float, default='0.5', help='Specify the confidence thresold for NMS')
parser.add_argument('--iou-thres', type=float, default='0.5', help='Specify the IoU thresold for NMS')

args = parser.parse_args()
split_set = args.split
size = args.imgsz
conf_thres = args.conf_thres
iou_thres = args.iou_thres

print(f"{size=}, {conf_thres=}, {iou_thres=}")


image_paths = glob(join(_CURRENT_DIR, '..', 'dataset', f'WIDER_{split_set}', 'images', '**', '*.jpg'), recursive=True)
assert image_paths != [], "We couldn't find any image !"
print(f"{len(image_paths)} image(s) are found!")
image_paths.sort()

_model = get_trained_model(size)

def get_bboxes(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = (img - np.array([104, 117, 123])).transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0)
    
    boxes, scores = _model(img)
    
    mask = scores > conf_thres
    print(scores.shape, mask.sum())
    print(scores.max(), scores.min())

    boxes = boxes[mask]
    scores = scores[mask]
    
    print(scores.shape, mask.sum())

    inds = batched_nms(boxes, scores, torch.tensor([0 for _ in range(scores.shape[0])]), iou_threshold=iou_thres)

    return boxes[inds].int().numpy()


cv.namedWindow("Validation", cv.WINDOW_GUI_NORMAL)

with tqdm(total=len(image_paths)) as pbar:
    for img_path in image_paths:  
        # read image
        original = cv.imread(img_path)
        original = cv.resize(original, (size, size))
        img = original.copy()
        
        # read bounding-boxes
        labels = get_bboxes(img)
        
        # show infos
        pbar.set_postfix({'image':split(img_path)[1], 'n_labels':labels.shape[0]})
        
        # visualize boxes
        for label in labels:
            x1, y1, x2, y2 = label
            cv.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=10)

        # show image
        cv.imshow('Validation', np.concatenate((original, img), axis=1))

        # wait for response
        _exit = False
        while not _exit:
            key = cv.waitKey(0)
            if key & 0xFF == ord('n'):
                pass
            elif key & 0xFF == ord('q'):
                cv.destroyAllWindows()
                print("Quitting ...")
                exit(0)
            else:
                print("Weird behavior: Type either y, n or d.")
                continue
            _exit = True

        pbar.update(1)


