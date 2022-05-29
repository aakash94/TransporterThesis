import gym
import numpy as np
from collections import deque
from stable_baselines3.common.monitor import Monitor


class FrameStackWrapper(gym.ObservationWrapper):

    def __init__(self, env, frame_stack_count=4, convert_greyscale=False):
        super().__init__(env)
        self.convert_greyscale = convert_greyscale
        self.env = env
        self.frames = deque(maxlen=frame_stack_count)
        state = env.reset()
        for i in range(frame_stack_count):
            self.frames.append(state)

    def observation(self, obs):
        if self.convert_greyscale:
            # TODO: Merge 3 channels of colour into greyscale
            pass
        self.frames.append(obs)
        state = self.frames_to_state()
        # print("obs ", obs.shape)
        # print("state ", type(state))
        return state

    def frames_to_state(self):
        # print("len", len(self.frames))
        # print("frame shape", self.frames[0].shape)
        state = np.concatenate(list(self.frames), axis=2)
        # print("mod state", state.shape)
        return state


if __name__ == "__main__":
    # from gym import envs
    # print(envs.registry.all())
    env_name = "CarRacing-v1"
    env = gym.make(env_name)
    env = Monitor(env)
    f = FrameStackWrapper(env=env)
    s = f.reset()
