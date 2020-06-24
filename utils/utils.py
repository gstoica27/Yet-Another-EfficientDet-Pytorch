# Author: Zylo117

import os

import cv2
import numpy as np
import torch
from glob import glob
from torch import nn
from torchvision.ops import nms
from torchvision.ops.boxes import batched_nms
from typing import Union
import uuid
import pickle

from utils.sync_batchnorm import SynchronizedBatchNorm2d

from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
import math
import webcolors

def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,


def preprocess(image_paths, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    if isinstance(image_paths, list):
        ori_imgs = []
        for image_path in image_paths:
            print(f'Imag to be read: {image_path}')
            img_data = cv2.imread(image_path)
            ori_imgs.append(img_data)
    else:
        ori_imgs = [cv2.imread(img_path) for img_path in [image_paths]]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def preprocess_video(*frame_from_video, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = frame_from_video
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out


def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])


def replace_w_sync_bn(m):
    for var_name in dir(m):
        target_attr = getattr(m, var_name)
        if type(target_attr) == torch.nn.BatchNorm2d:
            num_features = target_attr.num_features
            eps = target_attr.eps
            momentum = target_attr.momentum
            affine = target_attr.affine

            # get parameters
            running_mean = target_attr.running_mean
            running_var = target_attr.running_var
            if affine:
                weight = target_attr.weight
                bias = target_attr.bias

            setattr(m, var_name,
                    SynchronizedBatchNorm2d(num_features, eps, momentum, affine))

            target_attr = getattr(m, var_name)
            # set parameters
            target_attr.running_mean = running_mean
            target_attr.running_var = running_var
            if affine:
                target_attr.weight = weight
                target_attr.bias = bias

    for var_name, children in m.named_children():
        replace_w_sync_bn(children)


class CustomDataParallel(nn.DataParallel):
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, num_gpus):
        super().__init__(module)
        self.num_gpus = num_gpus

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus

        if splits == 0:
            raise Exception('Batchsize must be greater than num_gpus.')

        return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
                 inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True))
                for device_idx in range(len(devices))], \
               [kwargs] * len(devices)


def get_last_weights(weights_path):
    weights_path = glob(weights_path + f'/*.pth')
    weights_path = sorted(weights_path,
                          key=lambda x: int(x.rsplit('_')[-1].rsplit('.')[0]),
                          reverse=True)[0]
    print(f'using weights {weights_path}')
    return weights_path


def init_weights(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            if "conv_list" or "header" in name:
                variance_scaling_(module.weight.data)
            else:
                nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                if "classifier.header" in name:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    torch.nn.init.constant_(module.bias, bias_value)
                else:
                    module.bias.data.zero_()


def variance_scaling_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)


STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua','Beige', 'Azure','BlanchedAlmond','Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def from_colorname_to_bgr(color):
    rgb_color=webcolors.name_to_rgb(color)
    result=(rgb_color.blue,rgb_color.green,rgb_color.red)
    return result


def standard_to_bgr(list_color_name):
    standard= []
    for i in range(len(list_color_name)-36): #-36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard


def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index


def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)),0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0]+s_size[0]+15, c1[1] - t_size[1] -3
        cv2.rectangle(img, c1, c2 , color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0],c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

        
color_list = standard_to_bgr(STANDARD_COLORS)

# George's Image Region Creation Code
def create_image_region(bboxes, roi_bbox, filter_fn='threshold', filter_bound=.1, relevance_fn='iou', pad_amount=100):
    """
    Create image region around detected object using convex hull of surrounding boxes (including RoI bbox)
    :param bboxes: list of all detected bboxes in the image
    :type bboxes: np.array(dim=2): [N, 4]
    :param roi_bbox: RoI to create image region over. This will be either anomaly or context center
    :type roi_bbox: np.array([1, 4])
    :param filter_fn: filter method to choose relevant detections to RoI
    :type filter_fn: basestring
    :param filter_bound: bound for filtering
    :type filter_bound: float or int as make sense for method type
    :param relevance_fn: method to compute detection relevance to RoI
    :type relevance_fn: basestring
    :param minimum_size: lower bound for padding if desired
    :type minimum_size: float
    :return:
        1. bounding box of the convex hull of nearest bboxes
        2. all relevant bboxes (if any) to RoI
    :rtype:
        1. 1, 4] -> replacement for bbox which is fed to captioning model
        2. [RelevantSize, 4]
    """

    detection_overlaps = []
    for i, bbox in enumerate(bboxes):
        overlap = compute_overlap(bbox, roi_bbox, comp_type=relevance_fn, pad_amount=pad_amount)
        detection_overlaps.append(overlap)
    detection_overlaps = np.array(detection_overlaps)

    if filter_fn == 'threshold':
        valid_idxs = np.where(detection_overlaps >= filter_bound)[0]
        valid_bboxes = bboxes[valid_idxs]

    elif filter_fn == 'topk':
        # remove disjoint bboxes
        # argsort sorts in order of increasing. Put negative to sort in order of descending
        valid_idxs = np.argsort(-detection_overlaps)[:int(filter_bound)]
        valid_bboxes = bboxes[valid_idxs]

    elif filter_fn == 'iqr':
        # remove all bboxes with no overlaps, because they skew statistic toward 0, and we don't care about them
        valid_idxs = np.where(detection_overlaps > 0.0)[0]
        valid_bboxes = bboxes[valid_idxs]
        # filter out irrelevant bboxes
        valid_overlaps = detection_overlaps[valid_idxs]
        increasing_idxs = np.argsort(valid_overlaps)
        # order in terms of increasing relevance
        increasing_overlaps = valid_overlaps[increasing_idxs]
        increasing_boxes = valid_bboxes[increasing_idxs]
        # compute iqr for outlier extraction
        q1 = np.percentile(increasing_overlaps, 25, interpolation='midpoint')
        q3 = np.percentile(increasing_overlaps, 75, interpolation='midpoint')
        iqr = q3 - q1
        # compute outlier bound
        iqr_bound = np.median(increasing_overlaps) + 1.5 * iqr
        # get outlier bboxes (most well aligned with anomaly bbox)
        significant_idxs = np.where(increasing_overlaps >= iqr_bound)[0]
        # print('increasing bboxes: {}'.format(increasing_boxes.shape))
        # print('selected bboxes: {}'.format(increasing_boxes[significant_idxs].shape))
        valid_bboxes = increasing_boxes[significant_idxs]

    elif filter_fn == 'pad_solo':
        # pad all bboxes to be at least some size (so that they have surrounding context)
        padded_anomaly = pad_bbox(roi_bbox, pad_amount=pad_amount)
        valid_bboxes = padded_anomaly.reshape((1, 4))

    else:
        raise ValueError('filter_fn must be either: threshold, topk, iqr, pad_solo. It is now: {}'.format(filter_fn))
    # if filter_fn not in ['iqr', 'pad_solo']:
    #     valid_bboxes = bboxes[valid_idxs]

    # Add anomaly just in case
    valid_bboxes = np.concatenate((valid_bboxes.reshape((-1, 4)), roi_bbox.reshape((1, 4))), axis=0)
    convex_hull = compute_convex_hull(valid_bboxes)
    # Next few lines check an assertion that the convex hull is at least as large as the RoI, and if not saves data
    # for debugging
    convex_hull_area = compute_bbox_area(convex_hull)
    roi_bbox_area = compute_bbox_area(roi_bbox)

    if convex_hull_area < roi_bbox_area:
        data = {'bboxes': bboxes, 'anomaly': roi_bbox}
        with open('/home/scratch/gis/tmp_caption/captions/debug_data.pkl', 'w') as handle:
            pickle.dump(data, handle)
    if not bbox_in_hull(convex_hull, roi_bbox):
        data = {'bboxes': bboxes, 'anomaly': roi_bbox}
        with open('/home/scratch/gis/tmp_caption/captions/hull_debug_data.pkl', 'w') as handle:
            pickle.dump(data, handle)

    assert convex_hull_area >= roi_bbox_area, 'convex hull must be at least the area of the essential bbox'
    print('Convex Hull vsBBox prop: {}'.format(convex_hull_area / roi_bbox_area))
    assert bbox_in_hull(convex_hull, roi_bbox), 'hull must surround bbox'

    # return convex_hull and all valid bboxes
    return convex_hull, valid_bboxes


def bbox_in_hull(hull, bbox):
    if (hull[0] <= bbox[0] and hull[1] <= bbox[1] and hull[2] >= bbox[2] and hull[3] >= bbox[3]):
        return True
    return False


def compute_bbox_intersection_area(bbox1, bbox2):
    bbox1_x1 = bbox1[0]
    bbox1_y1 = bbox1[1]
    bbox1_x2 = bbox1[2]
    bbox1_y2 = bbox1[3]

    bbox2_x1 = bbox2[0]
    bbox2_y1 = bbox2[1]
    bbox2_x2 = bbox2[2]
    bbox2_y2 = bbox2[3]

    intersection_width = min(bbox1_x2, bbox2_x2) - max(bbox1_x1, bbox2_x1) + 1
    intersection_height = min(bbox1_y2, bbox2_y2) - max(bbox1_y1, bbox2_y1) + 1

    if intersection_height < 0 or intersection_width < 0:
        intersection_area = 0.
    else:
        intersection_area = intersection_width * intersection_height
    return intersection_area


def compute_overlap(bbox1, bbox2, comp_type='iou', pad_amount=10):
    """
    Compute iou between two bboxes
    :param bbox1: first bbox
    :type bbox1: np.array([1, 4])
    :param bbox2: second bbox
    :type bbox2: np.array([1, 4])
    :return: overlap proportion
    :rtype: float
    """
    # print('comp_type is: {} | equals giou? {}'.format(comp_type, comp_type == 'giou'))
    if comp_type == 'iou':
        iou = compute_iou(bbox1, bbox2)
    # area of intersection divided by candidate bounding box
    # this encourages the selection of condidate boxes that are near the anomaly
    elif comp_type == 'candidate':
        iou = compute_candidate(bbox1, bbox2)

    elif comp_type == 'hull':
        iou = compute_hull(bbox1, bbox2)

    elif comp_type == 'giou':
        iou = compute_giou(bbox1, bbox2)

    elif comp_type == 'pad_iou':
        # pad anomaly bbox
        padded_bbox2 = pad_bbox(bbox2, pad_amount=pad_amount)
        iou = compute_giou(bbox1, padded_bbox2)

    else:
        raise ValueError(
            'comp_type is: {} must be either iou: vanilla iou, candidate: prop intersect, hull: convex hull iou'.format(
                comp_type)
        )
    # print('bbox1: {} | bbox2: {}'.format(bbox1, bbox2))
    return iou


def compute_iou(bbox1, bbox2):
    intersection_area = compute_bbox_intersection_area(bbox1, bbox2)
    bbox1_area = compute_bbox_area(bbox1)
    bbox2_area = compute_bbox_area(bbox2)
    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)
    return iou


def compute_candidate(bbox1, bbox2):
    intersection_area = compute_bbox_intersection_area(bbox1, bbox2)
    bbox1_area = compute_bbox_area(bbox1)
    bbox2_area = compute_bbox_area(bbox2)
    iou = intersection_area / max(bbox1_area, bbox2_area)
    return iou


def compute_hull(bbox1, bbox2):
    intersection_area = compute_bbox_intersection_area(bbox1, bbox2)
    bbox1_area = compute_bbox_area(bbox1)
    bbox2_area = compute_bbox_area(bbox2)
    union_area = (bbox1_area + bbox2_area - intersection_area)
    stacked_bboxes = np.concatenate((bbox1.reshape((1, 4)), bbox2.reshape((1, 4))), axis=0)
    hull = compute_convex_hull(stacked_bboxes)
    hull_area = compute_bbox_area(hull)
    iou = union_area / hull_area
    return iou


def compute_giou(bbox1, bbox2):
    bbox1_area = compute_bbox_area(bbox1)
    bbox2_area = compute_bbox_area(bbox2)
    intersection_area = compute_bbox_intersection_area(bbox1, bbox2)
    union_area = (bbox1_area + bbox2_area - intersection_area)
    stacked_bboxes = np.concatenate((bbox1.reshape((1, 4)), bbox2.reshape((1, 4))), axis=0)
    hull = compute_convex_hull(stacked_bboxes)
    hull_area = compute_bbox_area(hull)

    hull_offset = (hull_area - union_area) / hull_area
    iou = intersection_area / union_area
    iou = iou - hull_offset
    return iou


def pad_bbox(bbox, pad_amount=10):
    """
    Pad bbox to a lower bound specified by minimum_size. Note, the radius (and padding) is measured along either the
    x or y axis, and the bbox radius is taken to be larger of the two. Padding is performed until that radius achieves
    the lower bound
    :param bbox: bbox to be padded
    :type bbox: [x1, y1, x2, y2]
    :param minimum_size: lower bound radius
    :type minimum_size: float
    :return: padded bbox
    :rtype: [1, 4]
    """
    bbox_ = bbox.copy()
    bbox_[0] -= pad_amount
    bbox_[1] -= pad_amount
    bbox_[2] += pad_amount
    bbox_[3] += pad_amount

    return bbox_


def compute_bbox_area(bbox):
    """
    compute bbox area
    :param bbox: bbox
    :type bbox: [1, 4]
    :return: area
    :rtype: float
    """
    # print(bbox)
    width = bbox[2] - bbox[0] + 1
    height = bbox[3] - bbox[1] + 1
    return width * height


def compute_convex_hull(bboxes):
    """
    Compute the convex hull over a series of bboxes
    :param bboxes: np array of bounding boxes
    :type bboxes: np.array([N, 4])
    :return: convex hull bounding box coordinates
    :rtype:
    """
    min_x1 = np.min(bboxes[:, 0])
    max_x2 = np.max(bboxes[:, 2])
    min_y1 = np.min(bboxes[:, 1])
    max_y2 = np.max(bboxes[:, 3])

    convex_hull = np.array([min_x1, min_y1, max_x2, max_y2])  # .reshape((1, 4))
    return convex_hull