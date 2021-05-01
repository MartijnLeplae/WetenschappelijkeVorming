from DQN_keras_rl import Trainer

for _ in range(10):
    trainer = Trainer('TreasureMap-v0')
    trainer.start()
    trainer.save_data()
