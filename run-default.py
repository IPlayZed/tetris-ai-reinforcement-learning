from training.trainer import Trainer

if __name__ == '__main__':
    trainer: Trainer = Trainer(verbosity=1)
    trainer.train(tensorboard_logname="1B-exp-lr-sched-def-reward")
    trainer.evaluate(video_episodes=100)
