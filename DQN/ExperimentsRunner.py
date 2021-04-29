from DQN_keras_rl import Trainer

# [n_states, bow, most_used, interval]
Parameters = [
    [False, True, False, False],  # bow
    [False, False, False, True],  # interval
    [False, True, False, True],   # interval-bow
    [True, True, False, False]    # states-bow
]
CODE = '121122'

for parameters in Parameters:
    for _ in range(10):
        trainer = Trainer('ButtonsWorld-v0')
        n_states, bow, most_used, interval = parameters
        trainer.env.set_user_parameters(n_states=n_states, bow=bow, most_used=most_used,
                                        interval=interval, code=CODE)
        trainer.start()
        trainer.save_data()
