from DQN_keras_rl import Trainer

# [n_states, bow, most_used, interval]
Parameters = [
    [False, True, False, False],  # bow
    [False, False, False, True],  # interval
    [False, True, False, True],   # interval-bow
    [True, False, False, False],  # states
    [True, True, False, False]    # states-bow
]
CODE = '2463154'  # '1232213'  # '1221112'  #   #    #   # '121122212'  # '121' # '121122'
#  BUTTONS = [(i+1) * 100 for i in range(2)]

for parameters in Parameters:
    for _ in range(10):
        trainer = Trainer('ButtonsWorld-v0')
        n_states, bow, most_used, interval = parameters
        trainer.env.set_user_parameters(n_states=n_states, bow=bow,
                                        most_used=most_used, interval=interval)
        trainer.start()
        trainer.save_data()
