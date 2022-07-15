import os
from Trainer import Trainer, write_log


def main():
    print("Hello")
    atari_env = False
    total_timesteps = 1000000
    eval_count = 10
    frame_stack_count = 4
    motion = False
    transporter = False

    if motion:
        # because not more than 2 frames are used.
        frame_stack_count = 2

    #################################
    frame_stack_count = 1
    motion = False
    transporter = False

    if motion:
        # because not more than 2 frames are used.
        frame_stack_count = 2

    t = Trainer(atari_env=atari_env, frame_stack_count=frame_stack_count, motion=motion, transporter=transporter)
    write_log(path=t.eval_path, string="mlp_1_80")
    print("All Set")
    # t.evaluate(n_eval_episodes=eval_count)
    t.train(total_timesteps=total_timesteps)
    print("Done Training")
    t.evaluate(n_eval_episodes=eval_count)
    t.save_model()
    ##############################################
    frame_stack_count = 4
    motion = False
    transporter = False

    if motion:
        # because not more than 2 frames are used.
        frame_stack_count = 2

    t = Trainer(atari_env=atari_env, frame_stack_count=frame_stack_count, motion=motion, transporter=transporter)
    write_log(path=t.eval_path, string="mlp_4_80")
    print("All Set")
    # t.evaluate(n_eval_episodes=eval_count)
    t.train(total_timesteps=total_timesteps)
    print("Done Training")
    t.evaluate(n_eval_episodes=eval_count)
    t.save_model()
    #############################################
    frame_stack_count = 2
    motion = True
    transporter = False

    if motion:
        # because not more than 2 frames are used.
        frame_stack_count = 2

    t = Trainer(atari_env=atari_env, frame_stack_count=frame_stack_count, motion=motion, transporter=transporter)
    write_log(path=t.eval_path, string="mlp_Motion_80")
    print("All Set")
    # t.evaluate(n_eval_episodes=eval_count)
    t.train(total_timesteps=total_timesteps)
    print("Done Training")
    t.evaluate(n_eval_episodes=eval_count)
    t.save_model()


if __name__ == "__main__":
    main()
