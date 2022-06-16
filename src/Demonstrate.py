import os
import gym
from stable_baselines3 import DQN
from Trainer import Trainer, get_env
from ThesisWrapper import ThesisWrapper
from stable_baselines3.common.evaluation import evaluate_policy


class Demonstrate():

    def __init__(self, atari_env=False, seed=42, verbose=0):
        frame_stack_count = 5
        model_name = "saved_model.zip"
        experiment_folder = "trial_expw" + str(frame_stack_count) + ""
        model_save_path = os.path.join(".", "models", experiment_folder, model_name)
        env = get_env(frame_stack_count=frame_stack_count, atari_env=atari_env, seed=seed)
        self.env = env
        self.model = DQN.load(path=model_save_path, env=self.env)

    def evaluate(self, n_eval_episodes=100):
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=n_eval_episodes)
        print("\nMean = ", mean_reward, "\t Std = ", std_reward)
        print()

    def demo_model(self, loop_count=10):
        for i in range(loop_count):
            total_reward = 0
            done = False
            state = self.env.reset()
            while not done:
                action, _states = self.model.predict(state)
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                self.env.render(mode="human")
            print("Iteration ", i, " Reward :", total_reward)

    def demo_random(self, loop_count=10):
        for i in range(loop_count):
            total_reward = 0
            done = False
            while not done:
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                self.env.render(mode="human")
            print("Iteration ", i, " Reward :", total_reward)

    def dump_frames(self, loop_count=100):
        pass


def main():
    d = Demonstrate()
    print("\n\nRandom!")
    d.demo_random(loop_count=5)
    print("\n\nModel!")
    d.demo_model(loop_count=5)


if __name__ == "__main__":
    main()
