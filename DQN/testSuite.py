import os
import traceback
from functools import reduce
from operator import mul

#import gym_two_rooms.envs
import envs
#from WetenschappelijkeVorming.DQN.DQN_keras_rl import Trainer
from DQN_keras_rl import Trainer
from numpy import arange

"""
A small program to test different environments 
with arrays of parameter values.

Each environment that is tested in the testSuite
should implement the method set_user_parameters. 
"""

counter = 0
total = 0


def get_input(name, floats=False):
    print(f'Enter NUMERAL inputs for {name} you want to check, separated by spaces.')
    if floats:
        lower, upper, step = map(float, input('Input form: range(<LOWER>, <UPPER>, <STEP-SIZE>) ').split()[:3])
    else:
        lower, upper, step = map(int, input('Input form: range(<LOWER>, <UPPER>, <STEP-SIZE>) ').split()[:3])
    return [x for x in arange(lower, upper, step)]


def get_boolean_input(name):
    print(f'BOOLEAN input for {name} you want to check.')
    options = [[True], [False], [True, False]]
    options_formatted = '\n'.join([f'({options.index(x)}) {x}' for x in options])
    choice = int(input('Select the number of the desired choice:\n' + options_formatted + '\n'))
    return options[choice]


def get_repr_length():
    return int(input("What is the desired length of the states/interval vector?"))


def handle_two_rooms(rooms_trainer: Trainer):
    """
    For the two rooms environment, we need the following parameters:
        - toggle,
        - step size
    """

    toggle_values = get_input("toggle values", floats=True)
    step_sizes = get_input("step sizes", floats=False)
    assert 0 not in step_sizes, "Step size can't be 0"
    start_counter(len(toggle_values) * len(step_sizes))
    for step_size in step_sizes:
        for toggle_value in toggle_values:
            rooms_trainer.env.reset()
            rooms_trainer.env.set_user_parameters(TOGGLE=toggle_value, step_size=step_size)
            try:
                trainer.start(save=True)
                trainer.plot(save=True)
                trainer.save_data()
            except ValueError:
                increment_counter()
                continue
    print_counter()


def handle_treasure_map(treasure_trainer: Trainer):
    """
    For the treasure map environment, we need the following parameters:
        - toggle,
        - step size
    """

    options = [False, True]
    use_in_place_repr = options[int(input('Do you want to use a constant repr size? \n(0) False\n(1) True\n'))]

    if use_in_place_repr:
        # nb_BOW_states_values = get_input("toggle values", floats=True)
        nb_BOW_states_values = input("How many of NB_PREV_STATES should be used for BOW? Multiple values should be "
                                     "separated by spaces:").split()
        nb_BOW_states_values = [int(x) for x in nb_BOW_states_values]

        step_sizes = get_input("step sizes", floats=False)
        assert 0 not in step_sizes, "Step size can't be 0"

        start_counter(len(nb_BOW_states_values) * len(step_sizes))
        for step_size in step_sizes:
            for bow_value in nb_BOW_states_values:
                treasure_trainer.env.reset()
                treasure_trainer.env.set_user_parameters(nb_BOW_states=bow_value, step_size=step_size)
                try:
                    trainer.name = f"({trainer.N_EPISODES}){trainer.env.get_name()}"
                    trainer.start(save=True)
                    trainer.plot(save=True)
                    trainer.save_data()
                except ValueError as e:
                    traceback.print_exception(type(e), e, e.__traceback__)
                    increment_counter()
                    continue
        print_counter()
    else:
        N_STATES = options[int(input('Boolean value for N_STATES: \n(0) False\n(1) True\n'))]
        BOW = options[int(input('Boolean value for BOW: \n(0) False\n(1) True\n'))]
        INTERVAL = options[int(input('Boolean value for INTERVAL: \n(0) False\n(1) True\n'))]
        MOST_USED = options[int(input('Boolean value for MOST_USED: \n(0) False\n(1) True\n'))]
        treasure_trainer.env.reset()
        treasure_trainer.env.set_user_parameters(N_STATES=N_STATES, BOW=BOW, INTERVAL=INTERVAL, MOST_USED=MOST_USED)
        try:
            trainer.name = f"({trainer.N_EPISODES}){trainer.env.get_name()}"
            trainer.start(save=True)
            trainer.plot(save=True)
            trainer.save_data()
        except ValueError as e:
            traceback.print_exception(type(e), e, e.__traceback__)


def handle_words_world(words_trainer: Trainer):
    """
    For the words world, we need the following boolean parameters:
        - add_states,
        - add_counts,
        - add_most_used,
        - add_interval
    """

    add_states_vals = get_boolean_input("add_states")
    add_counts_vals = get_boolean_input("add_counts")
    add_most_used_vals = get_boolean_input("add_most_used")
    add_interval_vals = get_boolean_input("add_interval")

    start_counter(len(add_states_vals) * len(add_counts_vals) * len(add_most_used_vals) * len(add_interval_vals))
    for state in add_states_vals:
        for count in add_counts_vals:
            for most_used in add_most_used_vals:
                for interval in add_interval_vals:
                    words_trainer.env.reset()
                    if not any([state, count, most_used, interval]):  # The history repr must be non-empty
                        continue
                    words_trainer.env.set_user_parameters(add_states=state, add_counts=count, add_most_used=most_used,
                                                          add_interval=interval)
                    try:
                        words_trainer.start(save=True)
                        words_trainer.plot(save=True)
                        words_trainer.save_data()
                    except ValueError as v:
                        print(v)
                        increment_counter()
                        continue
    print_counter()


def handle_buttons_world(buttons_trainer: Trainer):
    """
    For the buttons world we need the following boolean parameters:
        - N_STATES,
        - BOW,
        - MOST_USED,
        - INTERVAL
    """

    params = ["N_STATES", "BOW", "MOST_USED", "INTERVAL"]

    N_STATES_vals = get_boolean_input("N_STATES")
    BOW_vals = get_boolean_input("BOW")
    MOST_USED_vals = get_boolean_input("MOST_USED")
    INTERVAL_vals = get_boolean_input("INTERVAL")
    repr_length = 0
    if N_STATES_vals or INTERVAL_vals:
        repr_length = get_repr_length()

    # Total of counter is the product of lengths of lists of given values.
    start_counter(reduce(mul, map(len, [N_STATES_vals, BOW_vals, MOST_USED_vals, INTERVAL_vals]), 1))
    for state in N_STATES_vals:
        for count in BOW_vals:
            for most_used in MOST_USED_vals:
                for interval in INTERVAL_vals:
                    buttons_trainer.env.reset()
                    vals = [state, count, most_used, interval]
                    if not any(vals):  # The history repr must be non-empty
                        continue
                    # par_dic = {par: val for (par, val) in zip(params, vals)}  # using dict instead of **kwargs
                    print(f'[State:{state}, BOW:{count}, Most-Used:{most_used}, Interval:{interval}]')
                    buttons_trainer.env.set_user_parameters(n_states=state, bow=count, most_used=most_used,
                                                          interval=interval, nb_prev_states=repr_length)
                    try:
                        buttons_trainer.start(save=True)
                        buttons_trainer.plot(save=True)
                        buttons_trainer.save_data()
                    except ValueError as v:
                        print(v)
                        continue


def start_counter(nb_of_trainings):
    global counter, total
    counter = 0
    total = nb_of_trainings


def increment_counter():
    global counter
    counter += 1


def print_counter():
    global counter, total
    print("==============================================")
    print(f"====== {total - counter} out of {total} trainings succeeded. =======")
    print("==============================================")


def handle_cookie_domain(cookie_trainer: Trainer):
    # No possible parameters to choose as if right now
    raise NotImplemented


if __name__ == '__main__':
    possible_envs = ['ButtonsWorld-v0', 'TwoRooms-v0', 'WordsWorld-v0', 'TreasureMap-v0']
    # CookieDomain-v0 doesn't yet have adjustable parameters
    possible_modes = ['train', 'test']
    possible_envs_choices = '\n'.join([f"({possible_envs.index(env)}) {env}" for env in possible_envs])
    env_index = int(input('Index of environment to train in:\n' + possible_envs_choices + '\n'))
    env = possible_envs[env_index]
    trainer = Trainer(env)

    mode_index = int(input('Do you want to train (0) or test (1) the agent?\n'))
    mode = possible_modes[mode_index]
    if mode == "train":
        if env == "TwoRooms-v0":
            handle_two_rooms(trainer)
        elif env == "WordsWorld-v0":
            handle_words_world(trainer)
        elif env == "ButtonsWorld-v0":
            handle_buttons_world(trainer)
        elif env == "TreasureMap-v0":
            handle_treasure_map(trainer)
    elif mode == "test":
        filepath = "(500)States:3-231.h5"
        weights = input('Name of .h5-weights file: (leave empty if none)\n')
        if weights:
            filepath = weights
        dir_path = os.path.dirname(os.path.realpath(__file__))
        trainer.load_model(f'{dir_path}/models/{trainer.ENV}/{filepath}')
        trainer.test()
