from DQN_keras_rl import Trainer

# [n_states, bow, most_used, interval]
Parameters = [
    [False, True, False, False],  # bow
    [False, False, False, True],  # interval
    [False, True, False, True],   # interval-bow
    [True, True, False, False]    # states-bow
]
SEQUENCE = 121122

for parameters in Parameters:
    for _ in range(10):
        trainer = Trainer('ButtonsWorld-v0')
        trainer.env.set
