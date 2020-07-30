import skimage
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np
import json
import os
from tqdm import tqdm
from PIL import Image


def read_image(image_path):
    image = Image.open(image_path)
    return image

def save_crop(image, roi_bbox, save_path):
    region_crop = image.crop(([roi_bbox[0],
                               roi_bbox[1],
                               roi_bbox[2] + 1,
                               roi_bbox[3] + 1]))
    region_crop.save(save_path)

def save_regions_on_image(image, roi_regions, save_path):
    fig = plt.figure()
    plt.imshow(image)
    for roi_region in roi_regions:
        plt.gca().add_patch(plt.Rectangle(
            (roi_region[0], roi_region[1]),
            roi_region[2] - roi_region[0],
            roi_region[3] - roi_region[1],
            fill=False,
            edgecolor='b'))
    fig.savefig(save_path)
    plt.close()

os.chdir('/Users/georgestoica/Desktop/Research/Yet-Another-EfficientDet-Pytorch/datasets')
# frame2data_path = 'example/image_captions/01/frame2data_captions.pkl'
# frame2data_path = 'example/image_captions/01/image_captioning/fc_rl/frame2data_captions.pkl'
# frame2data_path = 'example/image_captions/01/full_images/frame2data_captions.pkl'
frame2data_path = '/Users/georgestoica/Desktop/frame2data.pkl'
# frame2data_path = 'example/image_captions/01/frame2data_captions.pkl'
frame2data = pickle.load(open(os.path.join(os.getcwd(), frame2data_path), 'rb'))
# frame2data = json.load(open(frame2data_path, 'r'))
frame = '739'
frame_data = frame2data[frame]
image_dir = '/Users/georgestoica/Desktop/16'
image_path = os.path.join(image_dir, frame + '.jpg')
image_rois = frame_data['rois']

# image = skimage.io.imread(image_path)
# roi = np.maximum(np.array(image_rois[1]).astype(np.int), 0)
# cropped_image = image[roi[1]:roi[3], roi[0]:roi[2], :]
image = read_image(image_path)
roi = np.array(image_rois[1])
cropped_image = image.crop((roi[0], roi[1], roi[2] + 1, roi[3] + 1))
#
# print(frame_data['captions'][2])
fig = plt.figure()
plt.imshow(cropped_image)
# fig.savefig('example/image_captions/01/{}.jpg'.format(frame), dpi=fig.dpi)
plt.show()
plt.close()
# save_regions_on_image(image, roi, save_path='example/image_captions/01/{}.jpg'.format(frame))
save_dir = '/Users/georgestoica/Desktop/16_regions'
os.makedirs(save_dir, exist_ok=True)
for frame, data in tqdm(frame2data.items()):
    image_path = os.path.join(image_dir, frame + '.jpg')
    image = read_image(image_path)
    save_regions_on_image(image, data['rois'], os.path.join(save_dir, frame + '.jpg'))