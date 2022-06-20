import argparse
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from alt_baselines import make_atari


def main():
    parser = argparse.ArgumentParser(description='Generate Enduro trajectories.')
    parser.add_argument('--datadir', default='data')
    parser.add_argument('--num_steps', default=3000, type=int)
    parser.add_argument('--num_trajectories', default=100, type=int)
    parser.add_argument('--seed', default=4242, type=int)
    args = parser.parse_args()

    env_name = "ALE/Enduro-v5"

    num_trajectories = args.num_trajectories
    datadir = args.datadir
    num_steps = args.num_steps
    np.random.seed(args.seed)

    def make_env(env_id, num_steps):
        env = make_atari(env_id=env_id, max_episode_steps=num_steps)
        # env = WarpFrame(env, 80, 80, grayscale=False)
        return env

    env = make_env(env_id=env_name, num_steps=num_steps)
    obs = env.reset()
    print("Data will be saved to {}".format(datadir))
    with tqdm(total=num_trajectories * num_steps) as pbar:
        for n in range(num_trajectories):
            os.makedirs('{}/{}'.format(datadir, n), exist_ok=True)
            obs = env.reset()
            t = 0
            Image.fromarray(obs).save('{}/{}/{}.png'.format(datadir, n, t))
            images = []
            while True:
                obs, r, done, _ = env.step(env.action_space.sample())
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
