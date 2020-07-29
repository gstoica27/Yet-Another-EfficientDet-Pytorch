import pickle
import os
import numpy as np
from tqdm import tqdm

def comply_with_python2(path):
    filename = os.path.splitext(os.path.basename(path))[0] + '2.pkl'
    file_dir = os.path.dirname(path)
    save_path = os.path.join(file_dir, filename)
    data = pickle.load(open(path, 'rb'))
    print('saving to: {}'.format(save_path))
    assert path != save_path, "cannot overwrite original file"
    pickle.dump(data, open(save_path, 'wb'), protocol=2)


if __name__ == '__main__':
    partition = 'test'
    region_type = 'regions'

    frame2data_dir = '/home/scratch/gis/datasets/Avenue/extracted_{}/{}'.format(region_type, partition)

    for video_id in tqdm(os.listdir(frame2data_dir)):
        # Skip non partition frames
        if video_id in {'d4', 'list.txt'}:
            continue

        frame2data_path = os.path.join(frame2data_dir, video_id, 'frame2full.pkl')
        comply_with_python2(frame2data_path)

