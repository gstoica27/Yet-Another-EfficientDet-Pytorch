from utils import utils
import numpy as np

# This file contains all code necessary to create image regions, as well as merge them together
#   cohesively into output format to caption generator

def threshold_filter(bboxes, overlaps, minimum_overlap):
    valid_idxs = np.where(overlaps >= minimum_overlap)[0]
    return bboxes[valid_idxs]

def topk_filter(bboxes, overlaps, topk):
    valid_idxs = np.argsort(-overlaps)[:int(topk) + 1]
    return bboxes[valid_idxs]

def iqr_filter(bboxes, overlaps, filter_edge):
    # remove all bboxes with no overlaps, because they skew statistic toward 0,
    #   and we don't care about them
    valid_idxs = np.where(overlaps > 0.0)[0]
    valid_bboxes = bboxes[valid_idxs]
    valid_overlaps = overlaps[valid_idxs]
    # sort in increasing order for IQR statistics
    increasing_idxs = np.argsort(valid_overlaps)
    increasing_bboxes = valid_bboxes[increasing_idxs]
    increasing_overlaps = valid_overlaps[increasing_idxs]

    # compute iqr for outlier extraction
    q1 = np.percentile(increasing_overlaps, 25, interpolation='midpoint')
    q3 = np.percentile(increasing_overlaps, 75, interpolation='midpoint')
    iqr = q3 - q1

    iqr_threshold = q3 + 1.5 * iqr
    significant_idxs = np.where(increasing_overlaps >= iqr_threshold)[0]
    return increasing_bboxes[significant_idxs]

def filter_bboxes(bboxes, overlaps, filter_fn, filter_edge):
    if filter_fn == 'threshold':
        valid_bboxes = threshold_filter(bboxes=bboxes,
                                        overlaps=overlaps,
                                        minimum_overlap=filter_edge)
    elif filter_fn == 'topk':
        valid_bboxes = topk_filter(bboxes=bboxes,
                                   overlaps=overlaps,
                                   topk=filter_edge)
    elif filter_fn == 'iqr':
        valid_bboxes = iqr_filter(bboxes=bboxes,
                                  overlaps=overlaps,
                                  filter_edge=filter_edge)
    elif filter_fn == 'no_attributes':
        valid_bboxes = np.array([])
    else:
        raise ValueError('filter_fn must be either: threshold, topk, iqr, no_attributes.')

    return valid_bboxes

def resize_region(initial_region, minimum_size):
    """
    Make image region have area at least as large as minimum_size
        on each side (squarely expand)
    :param initial_region: current region bbox coordinates
    :type initial_region: np.array([]) | [1, 4]
    :param minimum_size: minimum size for resized region
    :type minimum_size: int
    :return: resized region cordinates
    :rtype: np.array([]) | [1, 4]
    """
    return utils.pad_bbox(bbox=initial_region, minimum_size=minimum_size).\
        reshape(4)

def create_image_region(roi_bbox, rest_bboxes, region_params):
    """
    Create image region around ROI using nearby bboxes along direction
        specified by region parameters
    :param roi_bbox: ROI bbox coordinates
    :type roi_bbox: np.array([]) | [1, 4]
    :param rest_bboxes: bbox coordinates of all nearby bboxes
    :type rest_bboxes: np.array([]) | [1, 4]
    :param region_params: parameters specifying region creation:
        - relevance_fn: function name specifying how to measure bbox
            overlap with ROI
        - filter_fn: function name describing how to uncover
            relevant bboxes to ROI
        - filter_edge: parameter specifying filtering bound. This could
            be topK, threshold, etc... depending on the filter_fn
        - minimum_size: minimum size desired for ultimate bbbox
    :type region_params: dict
    :return: image region coords & all non-ROI bboxes used to create region
    :rtype: np.array([]), np.array([]) | [1, 4], [-1, 4]
    """
    relevance_fn = region_params['relevance_fn']
    filter_fn = region_params['filter_fn']
    filter_edge = region_params['filter_edge']
    minimum_size = region_params['minimum_size']

    detection_overlaps = []
    for i, bbox in enumerate(rest_bboxes):
        overlap = utils.compute_overlap(bbox,
                                        roi_bbox,
                                        comp_type=relevance_fn,
                                        minimum_size=minimum_size)
        detection_overlaps.append(overlap)
    detection_overlaps = np.array(detection_overlaps)

    valid_bboxes = filter_bboxes(bboxes=rest_bboxes,
                                 overlaps=detection_overlaps,
                                 filter_fn=filter_fn,
                                 filter_edge=filter_edge)
    # Add anomaly just in case
    valid_bboxes = np.concatenate((valid_bboxes.reshape((-1, 4)),
                                   roi_bbox.reshape((1, 4))),
                                  axis=0)
    convex_hull = utils.compute_convex_hull(valid_bboxes)
    # Next few lines check an assertion that the convex hull is
    #   at least as large as the RoI, and if not saves data for
    #   debugging
    image_region = resize_region(initial_region=convex_hull,
                                 minimum_size=minimum_size)
    # print('Image region: {}'.format(image_region))
    image_region_area = utils.compute_bbox_area(image_region)
    roi_bbox_area = utils.compute_bbox_area(roi_bbox)
    # correctness checks
    if image_region_area < roi_bbox_area:
        import pdb;pdb.set_trace()
    assert image_region_area >= roi_bbox_area, \
        'Region must be at least the area of the essential bbox'
    # print('Region vsBBox prop: {}'.format(image_region_area / roi_bbox_area))
    assert utils.bbox_in_hull(convex_hull, roi_bbox), 'Region must surround bbox'
    # return image_region and all valid bboxes
    return image_region, valid_bboxes
