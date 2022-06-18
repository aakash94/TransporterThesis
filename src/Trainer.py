import os
import gym

from collections import defaultdict
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from ThesisWrapper import ThesisWrapper


def get_env(frame_stack_count, atari_env=False, seed=42):
    if atari_env:
        env_name = "ALE/Enduro-v5"
        env = gym.make(env_name)
    else:
        env_name = "CarRacing-v1"
        env = gym.make(env_name, continuous=False)
    print(env_name)
    env = Monitor(env)
    env = ThesisWrapper(env, history_count=frame_stack_count, convert_greyscale=True, seed=seed, env_name=env_name)
    return env


class Trainer:

    def __init__(self, atari_env=False, seed=42, verbose=0, frame_stack_count=5):
        model_name = "saved_model.zip"
        experiment_folder = "gg" + str(frame_stack_count) + ""
        logs_root = os.path.join(".", "logs", experiment_folder)
        self.model_save_path = os.path.join(".", "models", experiment_folder, model_name)
        model_str = "DQN"
        if atari_env:
            env_name = "ALE/Enduro-v5"
        else:
            env_name = "CarRacing-v1"

        env = get_env(frame_stack_count=frame_stack_count, atari_env=atari_env, seed=seed)
        self.eval_env = get_env(frame_stack_count=frame_stack_count, atari_env=atari_env, seed=seed + 1)

        # https://github.com/hill-a/stable-baselines/issues/1087
        # self.eval_callback = EvalCallback(eval_env,best_model_save_path=self.model_save_path,eval_freq=100000,deterministic=True,render=False)

        self.env = env
        # self.video_recorder_callback = VideoRecorderCallback(env=eval_env, render_freq=500000, n_eval_episodes=4)

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
        print("Saved Model")

    def load_model(self):
        print("Loading Model")
        self.model = DQN.load(path=self.model_save_path)
        print("Loaded Model")

    def evaluate(self, n_eval_episodes=100):
        mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=n_eval_episodes)
        print("\nMean = ", mean_reward, "\t Std = ", std_reward)
        print()


def main():
    atari_env = False
    total_timesteps = 5000000
    eval_count = 30

    t = Trainer(atari_env=atari_env)
    # t.evaluate(n_eval_episodes=eval_count)
    t.train(total_timesteps=total_timesteps)
    print("Done Training")
    t.evaluate(n_eval_episodes=eval_count)
    t.save_model()
    # t.demonstrate()
    t.load_model()
    # t.demonstrate()
    t.evaluate(n_eval_episodes=eval_count)


if __name__ == "__main__":
    main()
