from training.trainer import Trainer

if __name__ == '__main__':
    trainer: Trainer = Trainer(verbosity=1, save_path_base="models/this-is-the-actual-custom-reward-no-lr-gpu",
                               evaluation_videos_path_base="videos/this-is-the-actual-custom-reward-no-lr-gpu",
                               lr_scheduler=0.0003, default_reward=False)
    trainer.train(tensorboard_logname="1B-this-is-the-actual-custom-reward-no-lr-gpu",
                  save_frequency=1_000_000)
    trainer.evaluate(video_episodes=100)
