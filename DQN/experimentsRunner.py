from DQN_keras_rl import Trainer

# [n_states, bow, most_used, interval]
Parameters = {
    'bow1':          [False, True, False, False],  # bow
    'interval1':     [False, False, False, True],  # interval
    'interval-bow1': [False, True, False, True],   # interval-bow
    'states1':       [True, False, False, False],  # states
    'states-bow1':   [True, True, False, False]    # states-bow
}
CODE = '121'  # '121122212'  # '121122'  #   #

for name, parameters in Parameters.items():
    for _ in range(10):
        trainer = Trainer('TreasureMapHard-v0')
        n_states, bow, most_used, interval = parameters
        trainer.env.set_user_parameters(n_states=n_states, bow=bow, most_used=most_used,
                                        interval=interval, code=CODE)
        trainer.start()
        try:
            trainer.save_data(subdir=name)
        except OSError:
            trainer.save_data()
