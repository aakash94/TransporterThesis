import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image


def generate(num_samples=1,
             size=16, dx=5):
    im = np.zeros((num_samples, size, size), dtype=np.float32)
    imt = np.zeros((num_samples, size, size), dtype=np.float32)

    x = np.random.randint(size, size=(num_samples,))
    y = np.random.randint(size, size=(num_samples,))

    for di in range(-1, 2):
        for dj in range(-1, 2):
            im[np.arange(num_samples),
               np.clip(y + di, 0, size - 1),
               np.clip(x + dj, 0, size - 1)] = 1.
    # im[:, y][x] = 1.

    # dx = np.random.randint(0, 5, size=(num_samples))
    dx = np.ones((num_samples,), dtype=np.int) * dx

    for di in range(-1, 2):
        for dj in range(-1, 2):
            imt[np.arange(num_samples),
                (np.clip(y + di, 0, size - 1)),
                (np.clip(x + dx + dj, 0, size - 1))] = 1.

    return im, imt


def vis_sample(sample):
    im, imt = sample
    im = np.concatenate(im, 0)
    imt = np.concatenate(imt, 0)
    print(im.shape)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(im, cmap='gray')
    ax[1].imshow(imt, cmap='gray')
    for a in ax.flat:
        a.set_axis_off()


class Dataset(object):
    _meta_data_file = 'metadata.json'

    def __init__(self, root, transform=None):
        self._root = root
        self._transform = transform
        with open('{}/{}'.format(root, self._meta_data_file), 'rt') as inp:
            self._metadata = json.load(inp)

    @property
    def num_trajectories(self):
        return self._metadata['num_trajectories']

    @property
    def num_timesteps(self):
        return self._metadata['num_timesteps']

    def __len__(self):
        raise NotImplementedError

    '''
    def get_image(self, n, t):
        im = np.array(Image.open('{}/{}/{}.png'.format(self._root, n, t)))
        return im
    '''

    def __getitem__(self, idx):
        n, t, tp1 = idx
        imt = np.array(Image.open('{}/{}/{}.png'.format(self._root, n, t)))
        imtp1 = np.array(Image.open('{}/{}/{}.png'.format(self._root, n, tp1)))
        imt, imtp1 = self.trnsformed_images(imt, imtp1)
        if self._transform is not None:
            imt = self._transform(imt)
            imtp1 = self._transform(imtp1)
        return imt, imtp1

    '''
    def get_trajectory(self, idx):
        images = [np.array(Image.open('{}/{}/{}.png'.format(self._root, idx, t))) for t in range(self.num_timesteps)]
        return [self._transform(im) for im in images]
    '''

    def trnsformed_images(self, i1, i2):
        if not isinstance(i1, np.ndarray):
            i1 = np.array(i1)

        if not isinstance(i2, np.ndarray):
            i2 = np.array(i2)

        i1 = cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY)
        i2 = cv2.cvtColor(i2, cv2.COLOR_RGB2GRAY)
        motion_i = self.get_motion(img_new=i2, img_old=i1)
        reconstruct = self.apply_flow(flow=motion_i, prev_i=i1, next_i=i2)
        return reconstruct, i2

    def get_motion(self, img_new, img_old):
        flow = cv2.calcOpticalFlowFarneback(prev=img_old,
                                            next=img_new,
                                            flow=None,
                                            pyr_scale=0.5,
                                            levels=3,
                                            winsize=15,
                                            iterations=3,
                                            poly_n=5,
                                            poly_sigma=1.2,
                                            flags=0)
        '''
        The below code is used to show the flow
        # https://www.geeksforgeeks.org/python-opencv-dense-optical-flow/
        mask = np.zeros_like(img_old)
        mask = np.dstack((mask, mask, mask))
        # Sets image saturation to maximum
        mask[..., 1] = 255

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        # Opens a new window and displays the output frame
        cv2.imshow("dense optical flow", rgb)
        cv2.imshow("OG image", img_new)
        cv2.waitKey(0)
        '''
        return flow

    def get_index(self, v, fv, max_v=84):
        i = int(v + fv)
        i = max(0, min(i, max_v - 1))
        return i

    def apply_flow(self, flow: np.ndarray, prev_i: np.ndarray, next_i: np.ndarray):
        # https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
        y_val, x_val = prev_i.shape
        final_i = np.zeros_like(prev_i)
        for y in range(y_val):
            for x in range(x_val):
                flow_val = flow[y][x]
                final_i[y][x] = next_i[self.get_index(v=y, fv=flow_val[1], max_v=y_val)] \
                    [self.get_index(v=x, fv=flow_val[0], max_v=x_val)]
        return final_i


class Sampler(torch.utils.data.Sampler):

    def __init__(self, dataset):
        self._dataset = dataset

    def __iter__(self):
        while True:
            n = np.random.randint(self._dataset.num_trajectories)
            num_images = self._dataset.num_timesteps
            t_ind = np.random.randint(0, num_images - 20)
            tp1_ind = t_ind + np.random.randint(20)
            yield n, t_ind, tp1_ind

    def __len__(self):
        raise NotImplementedError
