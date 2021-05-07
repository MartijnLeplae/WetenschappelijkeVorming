from DQN_keras_rl import Trainer
import multiprocessing


def do_experiment(buttons_trainer: Trainer, state=False, most_used=False, bow=False, interval=False, seq=None, act_random=False, buttons=None, n_prev=3):
    buttons_trainer.env.reset()
    if seq is not None:
        buttons_trainer.env.code = seq
    if buttons is not None:
        buttons_trainer.env.buttons = buttons
    print(f'[STARTED: State:{state}, BOW:{bow}, Most-Used:{most_used}, Interval:{interval}]')
    buttons_trainer.env.set_user_parameters(n_states=state, bow=bow, most_used=most_used,
                                            interval=interval, nb_prev_states=n_prev, act_random=act_random)
    buttons_trainer.start(save=False)
    buttons_trainer.save_data()
    print(f'[FINISHED: State:{state}, BOW:{bow}, Most-Used:{most_used}, Interval:{interval}]')


settings = [
            [False, False, False, False],
            [True, False, False, False],
            [True, False, True, False],
            [False, False, True, False],
            [False, False, True, True],
            [False, False, False, True],
            ]


def main():
    # Construct a list of lists which are the arguments for the experiments.
    n_episodes = 500
    sequences = ['2463154', '1221112']  # '121', '121122','121122212'
    buttons = [[100, 200, 300, 400, 500, 600], [100, 200]]
    # act_random = True
    all_args = []
    n_runs = 5
    for j, seq in enumerate(sequences):
        for setting in settings:
            for i in range(n_runs):
                trainer = Trainer(env='ButtonsWorld-v0', n_episodes=n_episodes, seed=i)
                args = [trainer] + setting + [seq, buttons[j]]
                all_args.append(args)

    # Execute all the experiments using the Pool() to distribute the processes over all the available cores. You can
    # also specify a max number of cores to be used if you specify this in the Pool() object as such-> e.g.: Pool(5)
    with multiprocessing.Pool() as p:
        p.starmap(do_experiment, all_args)
        p.close()
        p.join()


if __name__ == "__main__":
    main()
