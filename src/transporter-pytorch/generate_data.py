import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from stable_baselines3 import DQN
from alt_baselines import get_car_env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model():
    print("Loading Model")
    model = DQN.load(path="./../models/WTF_1/generator_cr.zip", device=device)
    print("Loaded Model")
    return model

def main():
    parser = argparse.ArgumentParser(description='Generate Pong trajectories.')
    parser.add_argument('--datadir', default='data2')
    parser.add_argument('--num_steps', default=1000, type=int)
    parser.add_argument('--num_trajectories', default=100, type=int)
    parser.add_argument('--seed', default=4242, type=int)
    args = parser.parse_args()

    env_name = "CarRacing-v1"

    num_trajectories = args.num_trajectories
    datadir = args.datadir
    num_steps = args.num_steps
    np.random.seed(args.seed)

    def make_env(env_id, num_steps):
        env = get_car_env(env_id=env_id, max_episode_steps=num_steps)
        return env

    env = make_env(env_id=env_name, num_steps=num_steps)
    obs = env.reset()
    model = load_model()
    print("Data will be saved to {}".format(datadir))
    with tqdm(total=num_trajectories * num_steps) as pbar:
        for n in range(num_trajectories):
            os.makedirs('{}/{}'.format(datadir, n), exist_ok=True)
            obs = env.reset()
            t = 0
            Image.fromarray(obs).save('{}/{}/{}.png'.format(datadir, n, t))
            images = []
            while True:
                # action = env.action_space.sample()
                # obs = torch.from_numpy(obs).to(device)
                # obs = torch.from_numpy(obs)#.to(device)
                # obs = obs.permute(2, 0, 1)
                obs = np.moveaxis(obs, -1, 0)
                action, _states = model.predict(obs, deterministic=True)
                obs, r, done, _ = env.step(action)
                Image.fromarray(obs).save('{}/{}/{}.png'.format(datadir, n, t))
                images.append(obs)
                t += 1
                if done:
                    break
            pbar.update(num_steps)
        with open('{}/metadata.json'.format(datadir), 'w') as out:
            json.dump({
                'num_trajectories': num_trajectories,
                'num_timesteps': num_steps
            }, out)


if __name__ == '__main__':
    main()
