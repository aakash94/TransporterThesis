import os
from collections import defaultdict
import torch
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from ThesisWrapper import ThesisWrapper, WarpFrame


def get_env(frame_stack_count, atari_env=False, seed=42, motion=False, transporter=False):
    if atari_env:
        env_name = "ALE/Enduro-v5"
        env = gym.make(env_name)
    else:
        env_name = "CarRacing-v1"
        env = gym.make(env_name, continuous=False)
    print(env_name)
    env = Monitor(env)
    env = WarpFrame(env=env, grayscale=False)
    env = ThesisWrapper(env,
                        history_count=frame_stack_count,
                        convert_greyscale=True,
                        seed=seed,
                        motion=motion,
                        keypoint=transporter)
    return env


def write_log(path, string):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a+') as f:
        f.write(string)
        f.write("\n")
    print(string)


class Trainer:

    def __init__(self, atari_env=False, seed=42, verbose=0, frame_stack_count=4, motion=False, transporter=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_name = "saved_model.zip"
        experiment_folder = "Test_" + str(frame_stack_count) + ""
        logs_root = os.path.join(".", "logs", experiment_folder)
        self.model_save_path = os.path.join(".", "models", experiment_folder, model_name)
        model_str = "DQN"
        if atari_env:
            env_name = "ALE/Enduro-v5"
        else:
            env_name = "CarRacing-v1"

        env = get_env(frame_stack_count=frame_stack_count,
                      atari_env=atari_env,
                      seed=seed,
                      motion=motion,
                      transporter=transporter)
        self.eval_env = get_env(frame_stack_count=frame_stack_count,
                                atari_env=atari_env,
                                seed=seed + 1,
                                motion=motion,
                                transporter=transporter)

        # https://github.com/hill-a/stable-baselines/issues/1087
        # self.eval_callback = EvalCallback(eval_env,best_model_save_path=self.model_save_path,eval_freq=100000,deterministic=True,render=False)

        self.env = env
        # self.video_recorder_callback = VideoRecorderCallback(env=eval_env, render_freq=500000, n_eval_episodes=4)

        logs_root = os.path.join(logs_root, env_name)

        print("Model ", model_str)
        logs_root = os.path.join(logs_root, model_str)

        self.eval_path = ""

        if motion:
            self.eval_path = os.path.join(logs_root, "mtn", "evals.txt")
            logs_root = os.path.join(logs_root, "mtn", "")
        elif transporter:
            self.eval_path = os.path.join(logs_root, "trnsprtr", "evals.txt")
            logs_root = os.path.join(logs_root, "trnsprtr", "")
        else:
            self.eval_path = os.path.join(logs_root, "nmtn", "evals.txt")
            logs_root = os.path.join(logs_root, "nmtn", "")

        self.model = DQN(
            'CnnPolicy',
            self.env,
            tensorboard_log=logs_root,
            device=self.device,
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
        print("Evaluating")
        mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=n_eval_episodes)
        result_string = "\nMean = " + str(mean_reward) + "\t Std = " + str(std_reward)
        print(result_string)
        write_log(path=self.eval_path, string=result_string)
        print()


def main():
    atari_env = False
    total_timesteps = 1000000
    eval_count = 10
    frame_stack_count = 1
    motion = False
    transporter = False

    t = Trainer(atari_env=atari_env, frame_stack_count=frame_stack_count, motion=motion, transporter=transporter)
    print("All Set")
    # t.evaluate(n_eval_episodes=eval_count)
    t.train(total_timesteps=total_timesteps)
    print("Done Training")
    t.evaluate(n_eval_episodes=eval_count)
    t.save_model()
    # t.demonstrate()
    t.load_model()
    # t.demonstrate()
    # t.evaluate(n_eval_episodes=eval_count)


if __name__ == "__main__":
    main()
