from DQN_keras_rl import Trainer
import multiprocessing


def do_experiment(buttons_trainer: Trainer, state=False, most_used=False, bow=False, interval=False, n_prev=3):
    buttons_trainer.env.reset()
    print(f'[STARTED: State:{state}, BOW:{bow}, Most-Used:{most_used}, Interval:{interval}]')
    buttons_trainer.env.set_user_parameters(n_states=state, bow=bow, most_used=most_used,
                                            interval=interval, nb_prev_states=n_prev)
    buttons_trainer.start(save=True)
    buttons_trainer.save_data()
    print(f'[FINISHED: State:{state}, BOW:{bow}, Most-Used:{most_used}, Interval:{interval}]')


settings = [[True, False, False, False],
            [True, False, True, False],
            [False, False, True, False],
            [False, False, True, True],
            [False, False, False, True]]


def main():
    # jobs = []
    # for i in range(5):
    #     for setting in settings:
    #         trainer = Trainer(env='ButtonsWorld-v0')
    #         args = [trainer] + setting
    #         jobs.append(multiprocessing.Process(target=do_experiment, args=args))
    #     for job in jobs:
    #         job.start()

    # Construct a list of lists which are the arguments for the experiments.
    all_args = []
    for i in range(10):
        for setting in settings:
            trainer = Trainer(env='ButtonsWorld-v0')
            args = [trainer] + setting
            all_args.append(args)

    # Execute all the experiments using the Pool() to distribute the processes over all the available cores. You can
    # also specify a max number of cores to be used if you specify this in the Pool() object as such-> e.g.: Pool(5)
    with multiprocessing.Pool() as p:
        p.starmap(do_experiment, all_args)


if __name__ == "__main__":
    main()
