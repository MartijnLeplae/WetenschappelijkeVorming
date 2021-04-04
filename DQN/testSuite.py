import os
import gym_two_rooms.envs
import envs
from WetenschappelijkeVorming.DQN.DQN_keras_rl import Trainer
from numpy import arange


"""
A small program to test different environments 
with arrays of parameter values.

Each environment that is tested in the testSuite,
should implement the method set_user_parameters. 
"""


def get_input(name):
    print(f'Enter NUMERAL inputs for {name} you want to check, separated by spaces.')
    lower, upper, step = map(float, input('Input form: range(<LOWER>, <UPPER>, <STEP-SIZE>) ').split()[:3])
    return [x for x in arange(lower, upper, step)]


def get_boolean_input(name):
    print(f'BOOLEAN input for {name} you want to check.')
    options = [[True], [False], [True, False]]
    options_formatted = '\n'.join([f'({options.index(x)}) {x}' for x in options])
    choice = int(input('Select the number of the desired choice:\n' + options_formatted + '\n'))
    return options[choice]


def handle_two_rooms(rooms_trainer: Trainer):
    """
    For the two rooms environment, we need the following parameters:
        - toggle,
        - step size
    """

    toggle_values = get_input("toggle values")
    step_sizes = get_input("step sizes")
    for step_size in step_sizes:
        for toggle_value in toggle_values:
            rooms_trainer.env.reset()
            rooms_trainer.env.set_user_parameters({'TOGGLE': toggle_value, 'step_size': step_size})
            trainer.start(save=True)
            trainer.plot(save=True)


def handle_words_world(words_trainer: Trainer):
    """
    For the words world, we need the following boolean parameters:
        - add_states,
        - add_counts,
        - add_most_used,
        - add_interval
    """

    params = ["add_states", "add_counts", "add_most_used", "add_interval"]

    add_states_vals = get_boolean_input("add_state")
    add_counts_vals = get_boolean_input("add_counts")
    add_most_used_vals = get_boolean_input("add_most_used")
    add_interval_vals = get_boolean_input("add_interval")
    for state in add_states_vals:
        for count in add_counts_vals:
            for most_used in add_most_used_vals:
                for interval in add_interval_vals:
                    words_trainer.env.reset()
                    vals = [state, count, most_used, interval]
                    par_dic = {par: val for (par, val) in zip(params, vals)}
                    words_trainer.env.set_user_parameters(par_dic)
                    words_trainer.start(save=True)
                    words_trainer.plot(save=True)


def handle_barry_world(barry_trainer: Trainer):
    """
    For the barry world we need the following boolean parameters:
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
    for state in N_STATES_vals:
        for count in BOW_vals:
            for most_used in MOST_USED_vals:
                for interval in INTERVAL_vals:
                    barry_trainer.env.reset()
                    vals = [state, count, most_used, interval]
                    par_dic = {par: val for (par, val) in zip(params, vals)}
                    barry_trainer.env.set_user_parameters(par_dic)
                    barry_trainer.start(save=True)
                    barry_trainer.plot(save=True)


def handle_cookie_domain(cookie_trainer: Trainer):
    # No possible parameters to choose as if right now
    raise NotImplemented


if __name__ == '__main__':
    possible_envs = ['BarryWorld-v0', 'TwoRooms-v0', 'CookieDomain-v0', 'WordsWorld-v0']
    possible_modes = ['train', 'test']
    possible_envs_choices = '\n'.join([f"({possible_envs.index(env)}) {env}" for env in possible_envs])
    env_index = int(input('Index of environment to train in:\n' + possible_envs_choices + '\n'))
    env = possible_envs[env_index]
    trainer = Trainer(env)

    mode_index = int(input('Do you want to train (0) or test (1) the agent? '))
    mode = possible_modes[mode_index]
    if mode == "train":
        if env == "TwoRooms-v0":
            handle_two_rooms(trainer)
        elif env == "WordsWorld-v0":
            handle_words_world(trainer)
        elif env == "BarryWorld-v0":
            handle_barry_world(trainer)

    if mode == "train":
        trainer.start(save=True)
        trainer.plot(save=True)
    elif mode == "test":
        filepath = "(500)States:3-231.h5"
        weights = input('Name of .h5-weights file: (leave empty if none) ')
        if weights:
            filepath = weights
        dir_path = os.path.dirname(os.path.realpath(__file__))
        trainer.load_model(f'{dir_path}/models/{trainer.ENV}/{filepath}')
        trainer.test()
