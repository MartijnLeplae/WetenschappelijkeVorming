import os

from DQN_keras_rl import Trainer

# [n_states, bow, most_used, interval]
parameters = [False, False, False, False, False]
# CODE = '121'  # '121122212'  # '121122'  #   #

for i in range(10):
    trainer = Trainer('TreasureMapHard-v0', seed=i)
    n_states, bow, most_used, interval, hist_sum = parameters
    trainer.env.set_user_parameters(N_STATES=n_states, BOW=bow, MOST_USED=most_used,
                                    INTERVAL=interval, HIST_SUM=hist_sum)
    if type(trainer.env.observation_space) == int:
        trainer.state_size = (trainer.env.observation_space,)  # (self.env.observation_space.n,)  #
    else:
        trainer.state_size = (trainer.env.observation_space.n,)
    trainer.start()
    try:
        trainer.save_data(subdir="30NoMemory")
    except OSError:
        trainer.save_data()
