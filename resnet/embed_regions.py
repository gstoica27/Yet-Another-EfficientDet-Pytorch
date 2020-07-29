import resnet.resnet
from resnet.resnet_utils import *
import os
import pickle
from tqdm import tqdm
import numpy as np
import skimage
import skimage.io
from torchvision import transforms as trn
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_resnet(path, use_cuda=False):
    cnn_model = 'resnet101'
    my_resnet = getattr(resnet.resnet, cnn_model)()
    my_resnet.load_state_dict(torch.load(path))
    my_resnet = myResnet(my_resnet)
    if use_cuda:
        my_resnet.cuda()
    my_resnet.eval()
    return my_resnet

def embed_region(model, image):
    embedding, attention = model(image)
    return embedding

def read_image(image_path):
    image = skimage.io.imread(image_path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.concatenate((image, image, image), axis=2)

    image = image.astype('float32') / 255.0
    image = torch.from_numpy(image.transpose([2, 0, 1])).cuda()
    image = preprocess(image)
    return image

def crop_image(image, roi):
    cropped_image = image[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
    return cropped_image

def embed_frame_regions(data, base_image, model):
    data['embeddings'] = []
    for roi in data['rois']:
        crop = crop_image(base_image, roi)
        embedding = embed_region(model, crop).detach().cpu().numpy().reshape(-1)
        assert embedding.shape[0] == 2048, f"embedding shape is: {embedding.shape}"
        data['embeddings'].append(embedding)
    data['embeddings'] = np.array(data['embeddings'])
    return data

def embed_video_regions(frame2data, video_dir, model):
    for frame, data in tqdm(frame2data.items()):
        frame_path = os.path.join(video_dir, frame + '.jpg')
        frame_image = read_image(frame_path)
        frame2data['frame'] = embed_frame_regions(data, frame_image, model)
    return frame2data

def save_frame2data(frame2data, path):
    with open(path, 'wb') as handle:
        pickle.dump(frame2data, handle)

def load_frame2data(path):
    return pickle.load(open(path, 'rb'))

if __name__ == '__main__':
    # Set arguments
    partition = 'test'
    use_cuda = True
    # if only object detection bboxes: 'detections', else: 'regions'
    region_type = 'detections'
    # Initialize paths
    resnet_path = '/home/scratch/gis/models/resnet/resnet101.pth'
    frame2data_dir = '/home/scratch/gis/datasets/Avenue/extracted_{}/{}'.format(region_type, partition)
    partition_dir = '/home/scratch/gis/datasets/Avenue/frames/{}'.format(partition)
    save_name = 'frame2full.pkl'

    model = load_resnet(resnet_path, use_cuda)

    for video_id in tqdm(os.listdir(partition_dir)):
        # Skip non partition frames
        if 'd4' in video_id:
            continue
        # Update paths
        frame2data_path = os.path.join(frame2data_dir, video_id, 'frame2data.pkl')
        video_dir = os.path.join(partition_dir, video_id)
        save_path = os.path.join(frame2data_dir, video_id, save_name)
        # Load extracted regions
        frame2data = load_frame2data(frame2data_path)
        # Extract region embeddings
        frame2full = embed_video_regions(frame2data, video_dir, model)
        # Save results
        print(f'saving to: {save_path}')
        save_frame2data(frame2full, save_path)