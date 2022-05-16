import os
import gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor


class Trainer():

    def __init__(self, atari_env=False, ppo_model=False, seed=42):

        # logs_root = os.path.join(".", "logs", "wrapped")
        logs_root = os.path.join(".", "logs", "baseline")
        if atari_env:
            env_name = "ALE/Enduro-v5"
        else:
            env_name = "CarRacing-v0"

        print("Env : ", env_name)
        self.env = Monitor(gym.make(env_name))
        logs_root = os.path.join(logs_root, env_name)

        if ppo_model:
            print("PPO Model")
            logs_root = os.path.join(logs_root, "PPO", "")

            self.model = PPO(
                'CnnPolicy',
                self.env,
                tensorboard_log=logs_root,
                # verbose=1,
                seed=seed)

        # elif a2c_model:
        #     print("A2C Model")
        #     logs_root = os.path.join(logs_root, "A2C", "")
        #
        #     self.model = A2C(
        #         'CnnPolicy',
        #         self.env,
        #         tensorboard_log=logs_root,
        #         # verbose=1,
        #         seed=seed)
        else:
            print("DQN Model")
            logs_root = os.path.join(logs_root, "DQN", "")

            self.model = DQN(
                'CnnPolicy',
                self.env,
                tensorboard_log=logs_root,
                # verbose=1,
                seed=seed)

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
    t = Trainer(a2c_model=True)
    t.train(total_timesteps=1000000)
    t.demonstrate()


if __name__ == "__main__":
    main()
