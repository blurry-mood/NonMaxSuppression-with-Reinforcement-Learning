from argparse import ArgumentParser
import numpy as np
from glob import glob
from os.path import join, split, splitext
import cv2 as cv
from tqdm.auto import tqdm

_CURRENT_DIR = split(__file__)[0]

parser = ArgumentParser(
    description="This script visualizes the WIDER Face dataset samples one at a time. To view the next image type `n` (next), and to quit type `q` (quit).")
parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                    help='Which dataset split to visualize: `train`, `val`, and `test`')
split_set = parser.parse_args().split


image_paths = glob(join(_CURRENT_DIR, '..', 'dataset', f'WIDER_{split_set}', 'images', '**', '*.jpg'), recursive=True)
assert image_paths != [], "We couldn't find any image !"
print(f"{len(image_paths)} image(s) are found!")
image_paths.sort()

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

cv.namedWindow("Validation", cv.WINDOW_GUI_NORMAL)

with tqdm(total=len(image_paths)) as pbar:
    for img_path in image_paths:  
        pbar.set_postfix({'image':split(img_path)[1]})
        
        # read image
        original = cv.imread(img_path)
        img = original.copy()

        try:
            # read bounding-boxes
            label_path = splitext(img_path)[0]+'.txt'
            labels = parse_label_file(label_path)
            
            # visualize boxes
            for label in labels:
                x,y,w,h = label
                x2,y2 = x+w, y+h
                cv.rectangle(img, (x, y), (x2, y2), color=(0, 0, 255), thickness=10)

            # show image
            cv.imshow('Validation', np.concatenate((original, img), axis=1))
        except:
            cv.imshow('Validation', original)

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


