import skimage
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np
import json
import os
from PIL import Image


def read_image(image_path):
    image = Image.open(image_path)
    return image

def save_regions_on_image(image, roi_bbox, save_path):
    region_crop = image.crop(([roi_bbox[0],
                               roi_bbox[1],
                               roi_bbox[2] + 1,
                               roi_bbox[3] + 1]))
    region_crop.save(save_path)

os.chdir('/Users/georgestoica/Desktop/Research/Yet-Another-EfficientDet-Pytorch/datasets')
# frame2data_path = 'example/image_captions/01/frame2data_captions.pkl'
frame2data_path = 'example/image_captions/01/image_captioning/fc_rl/frame2data_captions.pkl'
# frame2data_path = 'example/image_captions/01/full_images/frame2data_captions.pkl'
# frame2data_path = '/Users/georgestoica/Desktop/frame2data_captions.pkl'
# frame2data_path = 'example/image_captions/01/frame2data_captions.pkl'
frame2data = pickle.load(open(os.path.join(os.getcwd(), frame2data_path), 'rb'))
# frame2data = json.load(open(frame2data_path, 'r'))
frame = '453'
frame_data = frame2data[frame]
image_path = os.path.join(os.getcwd(), 'example/train/01/453.jpg')
image_rois = frame_data['rois']

image = skimage.io.imread(image_path)
roi = np.array(image_rois[2]).astype(np.int)
cropped_image = image[roi[1]-10:roi[3]+11, roi[0]-10:roi[2]+11, :]
# image = read_image(image_path)
# roi = np.array(image_rois[2])
# cropped_image = image.crop((roi[0], roi[1], roi[2] + 1, roi[3] + 1))

print(frame_data['captions'][2])
fig = plt.figure()
plt.imshow(cropped_image)
fig.savefig('example/image_captions/01/{}.jpg'.format(frame), dpi=fig.dpi)
plt.show()
plt.close()
# save_regions_on_image(image, roi, save_path='example/image_captions/01/{}.jpg'.format(frame))