from DQN_keras_rl import Trainer

for i in range(10):
    trainer = Trainer('TreasureMapHard-v0')
    trainer.start()
    trainer.save_data()
