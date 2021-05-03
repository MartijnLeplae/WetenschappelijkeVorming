from DQN_keras_rl import Trainer

for i in range(10):
    trainer = Trainer('TreasureMapHard-v0')
    n_states, bow, most_used, interval, hist_sum = [True, False, False, False, True]
    trainer.env.set_user_parameters(N_STATES=n_states, BOW=bow, MOST_USED=most_used, INTERVAL=interval, HIST_SUM=hist_sum)
    if type(trainer.env.observation_space) == int:
        trainer.state_size = (trainer.env.observation_space,)  # (self.env.observation_space.n,)  #
    else:
        trainer.state_size = (trainer.env.observation_space.n,)
    trainer.start()
    try:
        trainer.save_data(subdir='30/hist_sum1')
        print(f'N_STATES {trainer.env.N_STATES}, BOW: {trainer.env.BOW}, MU: {trainer.env.MOST_USED}, INTERVAL: {trainer.env.INTERVAL}, HIST_SUM: {trainer.env.HIST_SUM}')
    except OSError:
        trainer.save_data()
