import os
import gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor

from FrameStackWrapper import FrameStackWrapper
from ThesisWrapper import ThesisWrapper

PPO_str = "PPO"
A2C_str = "A2C"
DQN_str = "DQN"


class Trainer():

    def __init__(self, atari_env=False, model_str=PPO_str, seed=42, verbose=0):

        frame_stack_count = 5
        experiment_folder = "gfsw" + str(frame_stack_count) + ""
        logs_root = os.path.join(".", "logs", experiment_folder)
        # logs_root = os.path.join(".", "logs", "baseline")
        if atari_env:
            env_name = "ALE/Enduro-v5"
            env = gym.make(env_name)
        else:
            env_name = "CarRacing-v1"
            env = gym.make(env_name, continuous=False)

        print("Env : ", env_name)
        # env = gym.make(env_name)
        env = Monitor(env)
        # env = FrameStackWrapper(env, frame_stack_count=frame_stack_count, convert_greyscale=True)
        env = ThesisWrapper(env, history_count=frame_stack_count, convert_greyscale=True)
        self.env = env
        logs_root = os.path.join(logs_root, env_name)

        print("Model ", model_str)
        logs_root = os.path.join(logs_root, model_str, "")

        if model_str == PPO_str:
            self.model = PPO(
                'CnnPolicy',
                self.env,
                tensorboard_log=logs_root,
                verbose=verbose,
                seed=seed)

        elif model_str == A2C_str:
            self.model = A2C(
                'CnnPolicy',
                self.env,
                tensorboard_log=logs_root,
                verbose=verbose,
                seed=seed)

        elif model_str == DQN_str:
            self.model = DQN(
                'CnnPolicy',
                self.env,
                tensorboard_log=logs_root,
                buffer_size=50000,
                verbose=verbose,
                seed=seed)
        else:
            print("Invalid Model")

    def train(self, total_timesteps=1000000):
        print("Training")
        self.model.learn(total_timesteps=total_timesteps)

    def demonstrate(self, episode_count=10):
        for ep in range(episode_count):
            obs = self.env.reset()
            done = False
            while not done:
                action, _states = self.model.predict(obs)
                obs, rewards, done, info = self.env.step(action)
                self.env.render()


def main():
    atari_env = False
    total_timesteps = 5000000

    # t0 = Trainer(model_str=PPO_str, atari_env=atari_env)
    # t0.train(total_timesteps=total_timesteps)
    #
    # t1 = Trainer(model_str=A2C_str, atari_env=atari_env)
    # t1.train(total_timesteps=total_timesteps)

    t = Trainer(model_str=DQN_str, atari_env=atari_env)
    t.train(total_timesteps=total_timesteps)

    # t.demonstrate()
    # t1.demonstrate()
    # t2.demonstrate()


if __name__ == "__main__":
    main()
