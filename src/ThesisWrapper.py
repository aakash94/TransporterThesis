from collections import deque

import cv2
import gym
import gym.spaces as spaces
import numpy as np


class ThesisWrapper(gym.ObservationWrapper):

    def __init__(self,
                 env: gym.Env,
                 history_count=4,
                 motion=True,
                 convert_greyscale=True):
        super().__init__(env)
        self.history_count = history_count
        self.convert_greyscale = convert_greyscale
        self.frames = deque(maxlen=self.history_count)
        self.avg_image = None
        self.step_count = 0
        self.motion = motion
        state = self.reset()
        if isinstance(self.env.observation_space, gym.spaces.Box):
            self.observation_space = spaces.Box(low=0, high=255, shape=state.shape, dtype=np.uint8)

    def observation(self, obs):
        self.step_count += 1
        frame = self.operation_on_single_frame(obs=obs)
        self.frames.append(frame)
        state = self.operations_on_stack()
        return state

    def reset(self):
        self.step_count = 0
        og_state = self.env.reset()
        self.avg_image = None
        frame = self.operation_on_single_frame(obs=og_state)
        for i in range(self.history_count - 1):
            self.frames.append(frame)
        obs = self.observation(obs=og_state)
        return obs

    def operation_on_single_frame(self, obs):
        frame = obs
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
    env = ThesisWrapper(env=env, motion=True, convert_greyscale=True)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("step shape ", observation.shape)
    print("done checking")
