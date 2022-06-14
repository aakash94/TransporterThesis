import gym
import gym.spaces as spaces
import numpy as np
from collections import deque
import cv2
import os


class ThesisWrapper(gym.ObservationWrapper):

    def __init__(self,
                 env: gym.Env,
                 history_count=4,
                 dump_frames=False,
                 seed=42,
                 convert_greyscale=True):
        super().__init__(env)
        self.seed_val = seed
        np.random.seed(self.seed_val)
        self.history_count = history_count
        self.convert_greyscale = convert_greyscale
        self.dump_frames = dump_frames
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
        if self.convert_greyscale:
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return frame

    def operations_on_stack(self):
        # state = np.concatenate(list(self.frames), axis=2)
        state = np.dstack(list(self.frames))
        return state


if __name__ == "__main__":
    env_name = "CarRacing-v1"
    env = gym.make(env_name, continuous=False)
    env = ThesisWrapper(env=env)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("step shape ", observation.shape)
    print("done checking")
