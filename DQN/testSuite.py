import os
from WetenschappelijkeVorming.DQN.DQN_keras_rl import Trainer


def get_input(name):
    print(f'Enter input for {name} separated by spaces.')
    lower, upper, step = input('Input form: range(<LOWER>, <UPPER>, <STEP-SIZE>) ').split()[:3]
    return [x for x in range(lower, upper, step)]


def handle_two_rooms(two_rooms_trainer: Trainer):
    """
    For the two rooms environment, we need the following parameters:
        - toggle,
        - step size
    """

    toggle_values = get_input("toggle values")
    step_sizes = get_input("step sizes")
    for step_size in step_sizes:
        for toggle_value in toggle_values:
            two_rooms_trainer.env.set_user_parameters({'TOGGLE': toggle_value, 'step_size': step_size})
            trainer.start(save=True)
            trainer.plot(save=True)


def handle_cookie_domain(cookie_trainer: Trainer):
    pass


def handle_words_world(words_trainer: Trainer):
    pass


def handle_barry_world(barry_trainer: Trainer):
    pass


if __name__ == '__main__':
    possible_envs = ['BarryWorld-v0', 'TwoRooms-v0', 'CookieDomain-v0', 'WordsWorld-v0']
    possible_modes = ['train', 'test']
    possible_envs_formatted = '\n'.join([f"({possible_envs.index(env)}) {env}" for env in possible_envs])
    env_index = int(input('Index of environment to train in:\n' + possible_envs_formatted))
    env = possible_envs[env_index]
    trainer = Trainer(env)

    mode_index = int(input('Do you want to train (0) or test (1) the agent? '))
    mode = possible_modes[mode_index]
    if mode == "train":
        if env == "TwoRooms-v0":
            handle_two_rooms(trainer)

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

        # Example usage (when in ./WetenschappelijkeVorming/DQN directory):
        # python3 DQN_keras_rl.py -e BarryWorld-v0 -m test -w '(500)States:3-231.h5'
