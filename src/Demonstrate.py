import os
import gym
from stable_baselines3 import DQN
from ThesisWrapper import ThesisWrapper


class Demonstrate():

    def __init__(self, atari_env=False, seed=42, verbose=0):
        frame_stack_count = 5
        model_name = "saved_model.zip"
        experiment_folder = "trial_expw" + str(frame_stack_count) + ""
        # vexperiment_folder = "tw" + str(frame_stack_count) + ""
        model_save_path = os.path.join(".", "models", experiment_folder, model_name)

        if atari_env:
            env_name = "ALE/Enduro-v5"
            env = gym.make(env_name)
        else:
            env_name = "CarRacing-v1"
            env = gym.make(env_name, continuous=False)

        print("Env : ", env_name)
        env = ThesisWrapper(env, history_count=frame_stack_count, convert_greyscale=True)
        self.env = env
        self.model = DQN.load(path=model_save_path, env=self.env)

    def demo_model(self, loop_count=10):
        for i in range(loop_count):
            total_reward = 0
            done = False
            state = self.env.reset()
            while not done:
                action, _states = self.model.predict(state)
                # a = self.env.action_space.sample()
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
