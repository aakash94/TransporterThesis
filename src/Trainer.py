import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from ThesisWrapper import ThesisWrapper


class Trainer():

    def __init__(self, atari_env=False, seed=42, verbose=0):

        frame_stack_count = 5
        experiment_folder = "tw" + str(frame_stack_count) + ""
        logs_root = os.path.join(".", "logs", experiment_folder)
        model_save_path = os.path.join(".", "models", experiment_folder)
        model_str = "DQN"
        if atari_env:
            env_name = "ALE/Enduro-v5"
            env = gym.make(env_name)
            eval_env = gym.make(env_name)
        else:
            env_name = "CarRacing-v1"
            env = gym.make(env_name, continuous=False)
            eval_env = gym.make(env_name, continuous=False)

        print("Env : ", env_name)
        env = Monitor(env)

        env = ThesisWrapper(env, history_count=frame_stack_count, convert_greyscale=True)
        eval_env = ThesisWrapper(eval_env, history_count=frame_stack_count, convert_greyscale=True)

        # https://github.com/hill-a/stable-baselines/issues/1087
        self.eval_callback = EvalCallback(eval_env,
                                          best_model_save_path=model_save_path,
                                          eval_freq=100000,
                                          deterministic=True,
                                          render=False)

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
        self.model.learn(total_timesteps=total_timesteps, callback=self.eval_callback)

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

    t = Trainer(atari_env=atari_env)
    t.train(total_timesteps=total_timesteps)


if __name__ == "__main__":
    main()
