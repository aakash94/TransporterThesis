import gym
import gym.spaces as spaces
import numpy as np
from collections import deque


class FrameStackWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, frame_stack_count=4, convert_greyscale=False):
        super().__init__(env)
        self.convert_greyscale = convert_greyscale
        self.frame_stack_count = frame_stack_count
        self.frames = deque(maxlen=frame_stack_count)
        state = self.reset()
        if isinstance(self.env.observation_space, gym.spaces.Box):
            print("Correct Place")
            self.observation_space = spaces.Box(low=0, high=255, shape=state.shape, dtype=np.uint8)

    def observation(self, obs):
        if self.convert_greyscale:
            # TODO: Merge 3 channels of colour into greyscale
            pass
        self.frames.append(obs)
        state = self.frames_to_state()
        return state

    def frames_to_state(self):
        state = np.concatenate(list(self.frames), axis=2)
        return state

    def reset(self):
        og_state = self.env.reset()
        for i in range(self.frame_stack_count):
            self.frames.append(og_state)
        state = self.frames_to_state()
        return state


if __name__ == "__main__":
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_checker import check_env

    # from gym import envs
    # print(envs.registry.all())
    env_name = "CarRacing-v1"
    env = gym.make(env_name)
    env = Monitor(env)
    env = FrameStackWrapper(env=env)
    check_env(env=env)
    # s = env.reset()
    # print("reset shape ",s.shape)
    # action = env.action_space.sample()
    # observation, reward, done, info = env.step(action)
    # print("step shape ", observation.shape)

    print("done checking")
