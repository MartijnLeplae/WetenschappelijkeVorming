from DQN_keras_rl import Trainer

for i in range(10):
    trainer = Trainer('TreasureMap-v0')
    trainer.start(save=True)
    trainer.save_data()
