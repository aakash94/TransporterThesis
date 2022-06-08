import sys

import gym
import gym.spaces as spaces
import numpy as np
from collections import deque
import cv2


class ThesisWrapper(gym.ObservationWrapper):

    def __init__(self,
                 env: gym.Env,
                 history_count=4,
                 motion=False,
                 convert_greyscale=True):
        super().__init__(env)
        self.history_count = history_count
        self.convert_greyscale = convert_greyscale
        self.motion = motion
        self.frames = deque(maxlen=self.history_count)
        self.frame_count = 0  # TODO: Remove this
        self.running_average_image = None
        state = self.reset()
        if isinstance(self.env.observation_space, gym.spaces.Box):
            self.observation_space = spaces.Box(low=0, high=255, shape=state.shape, dtype=np.uint8)

    def observation(self, obs):
        self.frame_count += 1  # TODO: Remove this
        if self.frame_count == 1000:
            '''
            import pickle
            with open('frames.pickle', 'wb') as handle:
                pickle.dump(self.frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
            sys.exit("\n\n\n\nDONE!\n\n\n\n")
            '''
            pass

        frame = self.operation_on_single_frame(obs=obs)
        self.frames.append(frame)
        state = self.operations_on_stack()
        return state

    def reset(self):
        og_state = self.env.reset()
        frame = self.operation_on_single_frame(obs=og_state)
        self.running_average_image = np.float32(frame)  # For motion
        for i in range(self.history_count - 1):
            self.frames.append(frame)
        obs = self.observation(obs=og_state)
        self.frame_count = 0  # TODO: Remove this
        return obs

    def operation_on_single_frame(self, obs):
        frame = obs
        if self.convert_greyscale:
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return frame

    def operations_on_stack(self):
        if self.motion:
            state = self.get_motion_image()
            state = np.dstack(list(self.frames))
        else:
            # state = np.concatenate(list(self.frames), axis=2)
            state = np.dstack(list(self.frames))
        return state

    def get_motion_image(self):
        AVERAGE_ALPHA = 0.92  # 0-1 where 0 never adapts, and 1 instantly adapts
        MOVEMENT_THRESHOLD = 3  # Lower values pick up more movement
        MORPH_KERNEL = np.ones((2, 2), np.uint8)

        last_image = self.frames[-1]
        height, width, channels = 0, 0, 0
        gs_image = last_image
        if len(last_image.shape) == 2:
            height, width, = last_image.shape
            channels = 2
        elif len(last_image.shape) == 3:
            height, width, channels = last_image.shape
            gs_image = cv2.cvtColor(last_image, gs_image, cv2.COLOR_RGB2GRAY)

        img_mov = cv2.absdiff(self.frames[-2], gs_image)
        cv2.accumulateWeighted(img_mov, self.running_average_image, AVERAGE_ALPHA)
        background = cv2.convertScaleAbs(self.running_average_image)
        ret, th_img = cv2.threshold(background, MOVEMENT_THRESHOLD, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th_img, MORPH_KERNEL, iterations=3)
        allcontours, ret = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.frame_count > 900:
            cv2.imshow("Last", last_image)
            cv2.imshow("img_mov", img_mov)
            cv2.imshow("running average", self.running_average_image)
            cv2.imshow("threshold", th_img)
            cv2.imshow("dialated", dilated)
            cv2.waitKey(0)

        if self.frame_count == 1000:
            sys.exit("\n\n\n\n1000\n\n\n\n")
        # print("\n\n\n\n\n\n\n\nLast image shape", last_image.shape, "\n\n\n\n\n\n")



if __name__ == "__main__":
    env_name = "CarRacing-v1"
    env = gym.make(env_name, continuous=False)
    env = ThesisWrapper(env=env)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("step shape ", observation.shape)
    print("done checking")
