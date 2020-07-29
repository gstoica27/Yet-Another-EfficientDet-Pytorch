import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_image(image_path):
    image = Image.open(image_path)
    return image

def crop_image(image, roi_bbox):
    return image.crop([roi_bbox[0], roi_bbox[1], roi_bbox[2] + 1, roi_bbox[3] + 1])

if __name__ == '__main__':
    data_dir = '/Volumes/External HDD/Avenue'
    anomalies_path = os.path.join(data_dir, 'class_frequency_anomalies.pkl')
    frame_root_dir = os.path.join(data_dir, 'test')
    visualization_dir = os.path.join(data_dir, 'visuals')

    anomalies_info = pickle.load(open(anomalies_path, 'rb'))


    for video_id, anomaly_frames in tqdm(anomalies_info.items()):
        video_viz_dir = os.path.join(visualization_dir, video_id)
        os.makedirs(video_viz_dir, exist_ok=True)
        frame2num_obs = {}
        for anomaly_frame in tqdm(anomaly_frames):
            frame_id = anomaly_frame['frame_id']
            roi_bbox = anomaly_frame['region']
            anomaly_class = anomaly_frame['class']

            if frame_id not in frame2num_obs:
                frame2num_obs[frame_id] = 0
            else:
                frame2num_obs[frame_id] += 1
            num_obs = frame2num_obs[frame_id]
            viz_save_path = os.path.join(video_viz_dir, '{}_{}_{}.jpg'.format(
                frame_id, num_obs, anomaly_class
            ))

            frame_path = os.path.join(frame_root_dir, video_id, frame_id + '.jpg')
            frame = read_image(frame_path)
            fig = plt.figure()
            plt.imshow(frame)
            plt.gca().add_patch(plt.Rectangle(
                (roi_bbox[0], roi_bbox[1]),
                roi_bbox[2] - roi_bbox[0],
                roi_bbox[3] - roi_bbox[1],
                fill=False,
                edgecolor='r'))
            plt.savefig(viz_save_path)
            plt.close()
    # example_anomaly = anomalies_info['06'][0]
    # frame_path = os.path.join(
    #     frame_root_dir, example_anomaly['video_id'], example_anomaly['frame_id'] + '.jpg'
    # )
    # frame = read_image(frame_path)
    # roi_bbox = example_anomaly['region']
    # roi_crop = crop_image(image=frame, roi_bbox=roi_bbox)
    # fig = plt.figure()
    # plt.imshow(frame)
    # plt.gca().add_patch(plt.Rectangle(
    #     (roi_bbox[0], roi_bbox[1]),
    #     roi_bbox[2] - roi_bbox[0],
    #     roi_bbox[3] - roi_bbox[1],
    #     fill=False,
    #     edgecolor='r'))
    #
    # print(example_anomaly['class'])
    # # fig = plt.figure()
    # # plt.imshow(roi_crop)
    # plt.show()
    # plt.close()