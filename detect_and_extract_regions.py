# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors
import os
from backbone import EfficientDetBackbone
import cv2
import pickle
import numpy as np
from copy import deepcopy

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

from utils.region_creation import *


def get_name(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def display(preds, imgs, imshow=True, imwrite=False, write_dir=None):
    # for i in range(len(imgs)):
    for i, (name, img) in enumerate(imgs.items()):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(img, #imgs[i],
                         [x1, y1, x2, y2],
                         label=obj,
                         score=score,
                         color=color_list[get_index_label(obj, obj_list)])


        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs(write_dir, exist_ok=True)
            write_path = os.path.join(write_dir, f'{name}.jpg')
            cv2.imwrite(write_path, img)


def create_image_regions(detections, creation_schema):
    regions = []
    for frame_observations in detections:
        frame_regions = []
        frame_detections = frame_observations['rois']
        for frame_detection in frame_detections:
            frame_region, _ = create_image_region(roi_bbox=frame_detection,
                                                  rest_bboxes=frame_detections,
                                                  region_params=creation_schema)
            frame_regions.append(frame_region)
        regions.append(
            {
                'rois': frame_regions,
                'class_ids': frame_observations['class_ids'],
                'scores': frame_observations['scores']
            }
        )
    return regions


def get_frame2data(model_observations, frame_paths, frame_names):
    frame2data = {}
    triple_iter = zip(model_observations, frame_paths, frame_names)
    for model_observation, frame_path, frame_name in triple_iter:
        data = deepcopy(model_observation)
        data['path'] = frame_path
        frame2data[frame_name] = data
    return frame2data


def create_dir(*levels):
    level_dir = os.path.join(*levels)
    os.makedirs(level_dir, exist_ok=True)
    return level_dir

if __name__ == '__main__':

    compound_coef = 4
    force_input_size = None  # set None to use default size
    # Instantiate Schema
    creation_schema = {
        'relevance_fn': 'iou',
        'filter_fn': 'threshold',
        'filter_edge': .001,
        'minimum_size': 0
    }
    creation_name = '{}-{}-{}-{}'.format(
        creation_schema['relevance_fn'], creation_schema['filter_fn'],
        creation_schema['filter_edge'], creation_schema['minimum_size']
    )
    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    threshold = 0.2
    iou_threshold = 0.2

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']

    color_list = standard_to_bgr(STANDARD_COLORS)
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    cwd = os.getcwd()
    partition_dir = os.path.join(cwd, 'datasets/example/train')
    # img_filenames = ['453.jpg', '537.jpg', '946.jpg', '971.jpg']
    save_dir = os.path.join(cwd, 'datasets/example/extracted_regions/train')
    os.makedirs(save_dir, exist_ok=True)
    for video_id in os.listdir(partition_dir):
        video_dir = os.path.join(partition_dir, video_id)
        frame_filenames = os.listdir(video_dir)

        img_paths = [os.path.join(video_dir, img_name) for img_name in frame_filenames]
        img_names = [get_name(img_filepath) for img_filepath in img_paths]

        ori_imgs, framed_imgs, framed_metas = preprocess(img_paths, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=anchor_ratios, scales=anchor_scales)
        model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model = model.cuda()
        if use_float16:
            model = model.half()

        with torch.no_grad():
            features, regression, classification, anchors = model(x)
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)
        # {frame_id: frame_image, ...}
        imgs = dict([(name, img) for name, img in zip(img_names, ori_imgs)])
        # out: [{rois: [bbox1, bbox2, ..., bboxn], class_ids: [id1, ..., idn], scores: [score1, ..., scoren]}
        out = invert_affine(framed_metas, out)
        # Create Image Regions
        out = create_image_regions(detections=out, creation_schema=creation_schema)
        # Save Visualizations
        visual_save_dir = os.path.join(save_dir, f'd{compound_coef}', creation_name, video_id)
        display(out, imgs, imshow=False, imwrite=True, write_dir=visual_save_dir)
        # Save Extracted Pipeline Data
        frame2data = get_frame2data(model_observations=out, frame_paths=img_paths, frame_names=img_names)
        # pipeline_save_dir = os.path.join(save_dir, video_id)
        # os.makedirs(pipeline_save_dir, exist_ok=True)
        pipeline_save_dir = create_dir(save_dir, video_id)
        pipeline_save_path = os.path.join(pipeline_save_dir, 'frame2data.pkl')
        print(f'Saving information to: {pipeline_save_path}')
        with open(pipeline_save_path, 'wb') as handle:
            pickle.dump(frame2data, handle, protocol=2)
        print('Saved!')
