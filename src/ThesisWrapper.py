import os
from collections import deque

import cv2
import gym
import gym.spaces as spaces
import numpy as np
import torch

import transporter


def load_pointnet(model_path=""):
    np.random.seed(42)

    # Keep these params the same as when trained
    batch_size = 32
    image_channels = 3
    k = 4
    num_features = 32

    feature_encoder = transporter.FeatureEncoder(image_channels)
    pose_regressor = transporter.PoseRegressor(image_channels, k)
    refine_net = transporter.RefineNet(image_channels)

    model = transporter.Transporter(
        feature_encoder, pose_regressor, refine_net
    )

    model.load_state_dict(
        torch.load(model_path, map_location='cpu')
    )
    model.eval()
    pointnet = model.point_net
    # Pointnet here has the keypoints.
    # to view feature maps and all, following lines can be used
    # feature_maps = transporter.spatial_softmax(target_keypoints)
    # g map = transporter.gaussian_map(feature_maps, std)[idx, k_idx]
    return pointnet


class ThesisWrapper(gym.ObservationWrapper):

    def __init__(self,
                 env: gym.Env,
                 history_count=4,
                 dump_frames=False,
                 motion=False,
                 keypoint=True,
                 seed=42,
                 convert_greyscale=True):
        super().__init__(env)
        self.seed_val = seed
        np.random.seed(self.seed_val)
        self.history_count = history_count
        self.convert_greyscale = convert_greyscale
        self.dump_frames = dump_frames
        self.motion = motion
        self.keypoint = keypoint
        if self.keypoint:
            self.motion = False
        transporter_path = os.path.join(".", "models", "transporters", "model.pth")
        self.pointnet = load_pointnet(model_path=transporter_path)
        self.frames_dump_path = os.path.join(".", "frames", "dump")
        self.frames = deque(maxlen=self.history_count)
        state = self.reset()
        if isinstance(self.env.observation_space, gym.spaces.Box):
            self.observation_space = spaces.Box(low=0, high=255, shape=state.shape, dtype=np.uint8)

    def observation(self, obs):
        if self.dump_frames:
            pass
        frame = self.operation_on_single_frame(obs=obs)
        self.frames.append(frame)
        state = self.operations_on_stack()
        return state

    def reset(self):
        og_state = self.env.reset(seed=self.seed_val)
        frame = self.operation_on_single_frame(obs=og_state)
        for i in range(self.history_count - 1):
            self.frames.append(frame)
        obs = self.observation(obs=og_state)
        return obs

    def operation_on_single_frame(self, obs):
        frame = obs
        if self.keypoint:
            kps = self.pointnet(obs)
            return kps
        if self.convert_greyscale:
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return frame

    def operations_on_stack(self):
        if self.motion:
            img_motion = self.get_motion(img_new=self.frames[-1], img_old=self.frames[-2])
            state = np.dstack((self.frames[-1], img_motion))
        else:
            state = np.dstack(list(self.frames))
        return state

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


if __name__ == "__main__":
    env_name = "CarRacing-v1"
    env = gym.make(env_name, continuous=False)
    env = ThesisWrapper(env=env)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("step shape ", observation.shape)
    print("done checking")