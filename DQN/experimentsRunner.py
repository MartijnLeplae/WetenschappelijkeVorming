from DQN_keras_rl import Trainer

# [n_states, bow, most_used, interval]
Parameters = {
    'interval1':     [False, False, False, True],  # interval
    'bow1':          [False, True, False, False],  # bow
    'interval-bow1': [False, True, False, True],   # interval-bow
    'states1':       [True, False, False, False],  # states
    'states-bow1':   [True, True, False, False]    # states-bow
}
# CODE = '121'  # '121122212'  # '121122'  #   #

for name, parameters in Parameters.items():
    for _ in range(10):
        trainer = Trainer('TreasureMapHard-v0')
        n_states, bow, most_used, interval = parameters
        trainer.env.set_user_parameters(N_STATES=n_states, BOW=bow, MOST_USED=most_used,
                                        INTERVAL=interval)
        if type(trainer.env.observation_space) == int:
            trainer.state_size = (trainer.env.observation_space,)  # (self.env.observation_space.n,)  #
        else:
            trainer.state_size = (trainer.env.observation_space.n,)

        trainer.start()
        try:
            trainer.save_data(subdir=name)
        except OSError:
            trainer.save_data()
