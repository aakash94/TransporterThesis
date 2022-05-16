import gym


def dry_run(env_name="CartPole-v0", num_episodes=10):
    env = gym.make(env_name)
    print("type ", type(env))
    for i in range(num_episodes):
        observation = env.reset()
        rewards = 0
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            rewards += reward

        print("Episode Done")


def main():
    print("Testing Envs")
    env_name = "CarRacing-v0"
    #env_name = "Enduro-v4"
    #env = gym.make("ALE/Enduro-v5")
    dry_run(env_name=env_name)


if __name__ == "__main__":
    main()
