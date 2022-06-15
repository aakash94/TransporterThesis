import os
import gym

from collections import defaultdict
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from ThesisWrapper import ThesisWrapper

from typing import Callable


def get_env(rank: int, frame_stack_count: int = 4, atari_env: bool = False, seed=42):
    def _init() -> gym.Env:
        if atari_env:
            env_name = "ALE/Enduro-v5"
            env = gym.make(env_name)
        else:
            env_name = "CarRacing-v1"
            env = gym.make(env_name, continuous=False)
        env = ThesisWrapper(env, history_count=frame_stack_count, convert_greyscale=True, seed=seed + rank)
        return env

    set_random_seed(seed=seed)
    return _init()


def make_env(frame_stack_count: int = 4, atari_env: bool = False, seed=42):
    if atari_env:
        env_name = "ALE/Enduro-v5"
        env = gym.make(env_name)
    else:
        env_name = "CarRacing-v1"
        env = gym.make(env_name, continuous=False)
    env.seed(seed=seed)
    env = ThesisWrapper(env, history_count=frame_stack_count, convert_greyscale=True, seed=seed)
    return env


class Trainer:

    def __init__(self, atari_env=False, seed=42, verbose=0, num_cpu=4):

        frame_stack_count = 5
        experiment_folder = "u__" + str(frame_stack_count) + ""
        logs_root = os.path.join(".", "logs", experiment_folder)
        self.model_save_path = os.path.join(".", "models", experiment_folder, "saved_model.zip")
        model_str = "DQN"
        if atari_env:
            env_name = "ALE/Enduro-v5"
        else:
            env_name = "CarRacing-v1"
        print("Env : ", env_name)

        # env = get_env(frame_stack_count=frame_stack_count, atari_env=atari_env, seed=seed)
        self.eval_env = get_env(rank=0, frame_stack_count=frame_stack_count, atari_env=atari_env, seed=seed + 1)
        env = SubprocVecEnv([
            get_env(rank=i,
                    frame_stack_count=frame_stack_count,
                    atari_env=atari_env,
                    seed=seed) for i in range(num_cpu)
        ])
        env = VecMonitor(env)

        self.env = env
        logs_root = os.path.join(logs_root, env_name)

        print("Model ", model_str)
        logs_root = os.path.join(logs_root, model_str, "")

        self.model = DQN(
            'CnnPolicy',
            self.env,
            tensorboard_log=logs_root,
            buffer_size=50000,
            verbose=verbose,
            seed=seed)

    def train(self, total_timesteps=1000000):
        print("Training")
        # self.model.learn(total_timesteps=total_timesteps, callback=self.eval_callback)
        # self.model.learn(total_timesteps=total_timesteps, callback=self.video_recorder_callback)
        self.model.learn(total_timesteps=total_timesteps)

    def demonstrate(self, episode_count=10):
        action_count = defaultdict(int)
        for ep in range(episode_count):
            obs = self.env.reset()
            done = False
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, rewards, done, info = self.env.step(action)
                self.env.render()
                try:
                    action_count[action] += 1
                except:
                    print("\n\n\n\n\n\n\n\n\n\n\n\n\n!!!!!!!!!!!!!!\n")
                    print("Episode", episode_count)
                    print("Action", action)
                    print("type", type(action))
                    print("\n!!!!!!!!!!!!!!\n\n\n\n\n\n\n\n\n\n\n\n\n")
                finally:
                    pass

        print("\n\nAction Count\n", action_count)

    def save_model(self):
        print("Saving Model")
        self.model.save(path=self.model_save_path)
        print("Model Saved")

    def load_model(self):
        print("Loading Model")
        self.model = DQN.load(path=self.model_save_path)
        print("Model Loaded")


def main():
    atari_env = False
    total_timesteps = 20000

    t = Trainer(atari_env=atari_env)
    print("Model Created")
    mean_reward, std_reward = evaluate_policy(t.model, t.model.get_env(), n_eval_episodes=100)
    print("Mean: ", mean_reward, "Std :", std_reward)
    t.train(total_timesteps=total_timesteps)
    print("Done Training")
    mean_reward, std_reward = evaluate_policy(t.model, t.model.get_env(), n_eval_episodes=100)
    print("Mean: ", mean_reward, "Std :", std_reward)
    t.save_model()
    # t.demonstrate()
    t.load_model()
    # t.demonstrate()
    mean_reward, std_reward = evaluate_policy(t.model, t.eval_env, n_eval_episodes=100)
    print("Mean: ", mean_reward, "Std :", std_reward)


if __name__ == "__main__":
    main()
