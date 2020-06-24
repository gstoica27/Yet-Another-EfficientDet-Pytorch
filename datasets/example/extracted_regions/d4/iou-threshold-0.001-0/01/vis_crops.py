import skimage
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np
import json

# frame2data_path = '/Users/georgestoica/Desktop/Research/Yet-Another-EfficientDet-Pytorch/datasets/example/image_captions/01/frame2data_captions.pkl'
frame2data_path = '/Users/georgestoica/Desktop/Research/Yet-Another-EfficientDet-Pytorch/datasets/example/image_captions/01/frame2data_captions.json'

# frame2data = pickle.load(open(frame2data_path, 'rb'))
frame2data = json.load(open(frame2data_path, 'r'))
frame_data = frame2data['453']
image_path = '/Users/georgestoica/Desktop/Research/Yet-Another-EfficientDet-Pytorch/datasets/example/train/01/453.jpg'
image_rois = frame_data['rois']

image = skimage.io.imread(image_path)
print(image.shape)
roi_1 = np.array(image_rois[2]).astype(np.int)
cropped_image = image[roi_1[1]-10:roi_1[3]+11, roi_1[0]-10:roi_1[2]+11, :]
print(frame_data['captions'][2])
plt.imshow(cropped_image)
plt.show()