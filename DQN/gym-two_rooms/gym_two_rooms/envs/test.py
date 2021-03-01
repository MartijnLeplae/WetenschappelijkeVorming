import argparse

ENVS = ['cookie', 'rooms']
parser = argparse.ArgumentParser()


def check_env(env):
    if env in ENVS:
        print(env)
    else:
        print('Error: environment doesn\'t exist')


parser.add_argument('-e', '--environment', help="Specify an environment to train", type=check_env)
args = parser.parse_args()
arg = args.environment
print(arg)