import gym
from collections import deque
from stable_baselines3.common.monitor import Monitor

class FrameStackWrapper(gym.Wrapper):

    def __init__(self, env, frame_stack_count=4):
        super().__init__(env)
        self.env = env
        self.frames = deque(maxlen=frame_stack_count)
        print(len(self.frames))
        state = env.reset()
        for i in range(frame_stack_count):
            self.frames.append(state)
        print(len(self.frames))

    # TODO : Observation wrapper


if __name__ == "__main__":
    env_name = "CarRacing-v0"
    env = gym.make(env_name)
    env = Monitor(env)
    f = FrameStackWrapper(env=env)
