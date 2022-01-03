from glob import glob
from os.path import split, join, splitext
from tqdm.auto import tqdm

_CURRENT_DIR = split(__file__)[0]

sets = ['train', 'val']


def parse_txt(text: str):
    bboxes = {}
    lines = text.split('\n')
    try:
        lines.remove('')
    except:
        pass
    i = 0
    n = len(lines)
    while i < n-1:
        img_path = lines[i]
        n_boxes = max(1, int(lines[i+1]))
        B = []
        for b in range(n_boxes):
            x, y, w, h, *_ = map(int, lines[i+b+2].split())
            B.append(f'{x} {y} {w} {h}')
        bboxes[img_path] = B
        i += n_boxes + 2

    return bboxes


for set in sets:
    print(f"Proceeding with {set} set.")

    txt_file = open(join(_CURRENT_DIR, 'wider_face_split',
                         f'wider_face_{set}_bbx_gt.txt')).read()
    print("Bounding boxes file opened successfully !")

    bboxes = parse_txt(txt_file)
    print("Bounding boxes are successfully parsed !")

    for img_path, _bboxes in tqdm(bboxes.items(), total=len(bboxes)):
        with open(join(_CURRENT_DIR, f'WIDER_{set}', 'images', splitext(img_path)[0]+'.txt'), 'w') as f:
            # print(list(map(' '.join, _bboxes)))
            f.write('\n'.join(_bboxes))
    print(f"Processing of {set} set is done successfully!")