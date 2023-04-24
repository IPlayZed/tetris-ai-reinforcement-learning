from training.trainer import Trainer

if __name__ == '__main__':
    trainer: Trainer = Trainer(verbosity=1, save_path_base="models/1B-lr-exp-custom-reward-gpu",
                               evaluation_videos_path_base="videos/1B-lr-exp-custom-reward-gpu",
                               default_reward=True)
    trainer.train(tensorboard_logname="1B-lr-exp-custom-reward-gpu",
                  save_frequency=1_000_000)
    trainer.evaluate(video_episodes=100)
