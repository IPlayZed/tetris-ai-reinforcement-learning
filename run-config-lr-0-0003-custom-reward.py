from training.trainer import Trainer

if __name__ == '__main__':
    trainer: Trainer = Trainer(verbosity=1, save_path_base="models/1B-lr-0-0003-new-custom-reward",
                               evaluation_videos_path_base="videos/1B-lr-0-0003-new-custom-reward",
                               lr_scheduler=3e-4, default_reward=False)
    trainer.train(tensorboard_logname="1B-lr-0-0003-new-custom-reward",
                  save_frequency=1_000_000)
    trainer.evaluate(video_episodes=100)
