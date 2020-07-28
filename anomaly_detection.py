import os
import numpy as np
import pickle

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']

def get_id2class(class_list):
    id2class = {}
    for idx, class_name in enumerate(class_list):
        id2class[idx] = class_name
    return id2class

def compute_class_frequencies(partition_dir, id2class):
    class2freq = {}
    for video_id in os.listdir(partition_dir):
        frame2data_path = os.path.join(partition_dir, video_id, 'frame2data.pkl')
        frame2data = pickle.load(open(frame2data_path, 'rb'))
        for frame, data in frame2data.items():
            class_ids = data['class_ids']
            for class_id in class_ids:
                class_name = id2class[class_id]
                if class_name not in class2freq:
                    class2freq[class_name] = 0
                class2freq[class_name] += 1
    return class2freq

def compute_iqr(elements):
    ordered_elements = sorted(elements)
    # compute iqr for outlier extraction
    q1 = np.percentile(ordered_elements, 25, interpolation='midpoint')
    q3 = np.percentile(ordered_elements, 75, interpolation='midpoint')
    iqr = q3 - q1
    threshold = q1 - 1.5 * iqr
    return threshold

def compute_percentile(elements, fraction):
    return np.percentile(elements, int(fraction*100), interpolation='midpoint')

def compute_anomaly_bound(class2freq, metric='iqr'):
    frequencies = list(class2freq.values())
    if metric == 'iqr':
        bound = compute_iqr(frequencies)
    elif 'percentile' in metric:
        percentile = metric.split('_')[-1]
        bound = compute_percentile(frequencies, percentile)
    elif 'threshold' in metric:
        bound = metric.split('_')[-1]
    else:
        raise ValueError('metric must be in: {iqr, percentile_*, threshold_*}')
    return bound

def identify_anomalies(partition_dir, class2freqs, bound):
    anomalies_info = {}
    for video_id in os.listdir(partition_dir):
        # Skip non partition frames
        if 'd4' in video_id:
            continue
        frame2data_path = os.path.join(partition_dir, video_id, 'frame2data.pkl')
        frame2data = pickle.load(open(frame2data_path, 'rb'))
        for frame, data in frame2data.items():
            class_ids = data['class_ids']
            for idx, class_id in enumerate(class_ids):
                class_name = id2class[class_id]
                class_freq = class2freqs[class_name]
                # We've found an anomaly
                if class_freq <= bound:
                    anomaly_info = {
                        'video_id': video_id,
                        'frame_id': frame,
                        'class': class_name,
                        'region': data['rois'][idx]
                    }
                    if video_id not in anomalies_info:
                        anomalies_info[video_id] = []
                    anomalies_info[video_id].append(anomaly_info)
    return anomalies_info


if __name__ == '__main__':
    data_dir = '/home/scratch/gis/datasets/Avenue/extracted_regions'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    id2class = get_id2class(CLASSES)
    train_class2freqs = compute_class_frequencies(partition_dir=train_dir, id2class=id2class)
    anomaly_bound = compute_anomaly_bound(class2freq=train_class2freqs, metric='threshold_100')
    print('Class frequencies: {}'.format(train_class2freqs))
    print('Anomaly Bound: {}'.format(anomaly_bound))
    anomalies = identify_anomalies(test_dir, train_class2freqs, anomaly_bound)
    print('We have: {} anomalies'.format(len(anomalies)))
    save_path = os.path.join(data_dir, 'class_frequency_anomalies.pkl')
    print('saving anomalies to: {}'.format(save_path))
    pickle.dump(anomalies, open(save_path, 'wb'))
