from DQN_keras_rl import Trainer

for i in range(10):
    trainer = Trainer('TreasureMapHard-v0')
    trainer.env.set_user_parameters(N_STATES=True, BOW=False, MOST_USED=False, INTERVAL=False, HIST_SUM=True)
    if type(trainer.env.observation_space) == int:
        trainer.state_size = (trainer.env.observation_space,)  # (self.env.observation_space.n,)  #
    else:
        trainer.state_size = (trainer.env.observation_space.n,)
    trainer.start()
    trainer.save_data()
