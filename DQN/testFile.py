from DQN_keras_rl import Trainer

for i in range(10):
    trainer = Trainer('TreasureMap-v0')
    trainer.env.set_user_parameters(step_size=1, INTERVAL=False, MOST_USED=True, use_in_place_repr=False)
    trainer.start(save=True)
    trainer.save_data()
