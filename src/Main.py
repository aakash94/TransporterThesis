from Trainer import Trainer, write_log


def main():
    print("Hello")
    atari_env = False
    total_timesteps = 1000000
    eval_count = 10
    frame_stack_count = 10

    #################################
    atari_env = False
    motion = False
    transporter = False

    t = Trainer(atari_env=atari_env, frame_stack_count=frame_stack_count, motion=motion, transporter=transporter)
    write_log(path=t.eval_path, string="10_80")
    print("All Set")
    t.evaluate(n_eval_episodes=eval_count)
    t.train(total_timesteps=total_timesteps)
    print("Done Training")
    t.evaluate(n_eval_episodes=eval_count)
    t.save_model()
    ##############################################
    atari_env = True
    motion = False
    transporter = False

    t = Trainer(atari_env=atari_env, frame_stack_count=frame_stack_count, motion=motion, transporter=transporter)
    write_log(path=t.eval_path, string="10_80")
    print("All Set")
    t.evaluate(n_eval_episodes=eval_count)
    t.train(total_timesteps=total_timesteps)
    print("Done Training")
    t.evaluate(n_eval_episodes=eval_count)
    t.save_model()


if __name__ == "__main__":
    main()
