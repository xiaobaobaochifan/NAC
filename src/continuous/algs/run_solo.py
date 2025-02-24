import torch
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import time
import algs.core as core
from algs.costs import CostFunction
from algs.nac import offline_generation, nac_learn, state_init_generator
from algs.nac_eva import nac_evaluate
from algs.nac_dec import eva_old, nac_decision, updatability, eva_fixed
from utils.logx import EpochLogger
from algs.nac_test_online import online_eva

def run(env_name="Hopper-v4", offline_size=int(1e6), batch_size=256, scratch=False,
        act_partition='default', lc=5, tc=0, seed = 0, ac_kwargs=dict(hidden_sizes=[256]*2),
        state_gen_runs_trn=1, state_gen_runs_eva=10000, steps_per_epoch=1000,
        epochs_trn=1000, epochs_eva=10, gamma=0.99, lr=3e-4, act_gen_runs_trn=1000,
        act_gen_runs_eva=10000, logger_kwargs=dict(exp_name='rep1'), save_freq=1,
        num_test_episodes=10, max_ep_len=1000, q_grad_clip=1.0, pi_grad_clip=1.0,
        rate_stop=1.0, epochs_stop=10, upper_stop=50, lower_stop=10,
        start_optimal=False):
    
    # Initial setup
    start_time = time.time()
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env_fn = lambda: gym.make(env_name)
    env = env_fn()
    obs_space = Box(low=env.observation_space.low, high=env.observation_space.high,
                    shape=env.observation_space.shape, dtype=np.float32)
    act_space = Box(low=env.action_space.low, high=env.action_space.high,
                    shape=env.action_space.shape, dtype=np.float32)
    obs_dim = obs_space.shape
    act_dim = act_space.shape[0]
    actor_critic=core.MLPActorCritic
    if start_optimal:
        ac_old = torch.load(f'utils/optimals/{env_name}/pyt_save/model.pt')
    else:
        ac_old = actor_critic(obs_space, act_space, **ac_kwargs)
        if env_name == 'Ant-v4':
            ac_old = torch.load(f'utils/weak/{env_name}/pyt_save/model.pt')
    
    o_initial = state_init_generator(env_name=env_name, env_fn=env_fn)
    cost_class = CostFunction(env_fn, act_partition=act_partition,
                              state_gen_runs=state_gen_runs_trn,
                              lc=lc, tc=tc)
    cost_fn = cost_class.duo

    # Offline data initialization
    offline_data = offline_generation(env_fn, ac_old, offline_size=offline_size,
                                      start_steps=10000, max_ep_len=max_ep_len)
    
    logger.log_tabular('CollectTime', time.time()-start_time)
    timept_collect = time.time()

    # Offline evaluation of the given old policy
    ac_old = eva_old(env_fn, offline_data, ac_old, cost_fn,seed = seed,
                     steps_per_epoch=steps_per_epoch, epochs=epochs_eva,
                     gamma = gamma, polyak=0.995, lr=lr, batch_size=batch_size,
                     primary=True, logger_kwargs=logger_kwargs)
    old_ref = eva_fixed(ac_old, ac_old, o_initial, cost_fn,
                        act_gen_runs=act_gen_runs_eva, primary = True)
    old_ref = old_ref.item()
    # print(old_ref)

    # Switch-optimal candidate training
    ac_new = nac_learn(env_fn, ac_old, cost_fn, o_initial, offline_data, old_ref,
                       actor_critic=core.MLPActorCritic, ac_kwargs=ac_kwargs,
                       seed = seed, steps_per_epoch=steps_per_epoch,
                       epochs=epochs_trn, gamma = gamma, polyak=0.995, lr=lr,
                       batch_size=batch_size, act_gen_runs=act_gen_runs_trn,
                       logger_kwargs=logger_kwargs, save_freq=save_freq,
                       scratch=scratch, num_test_episodes=num_test_episodes,
                       max_ep_len=max_ep_len, q_grad_clip=q_grad_clip,
                       pi_grad_clip=pi_grad_clip, rate_stop=rate_stop,
                       epochs_stop=epochs_stop, upper_stop=upper_stop,
                       lower_stop=lower_stop)

    # # Follow-up offline evaluation of the resulting candidate new policy
    # ac_new = nac_evaluate(env_fn, offline_data, ac_old, ac_new, cost_fn,
    #                       seed = seed, steps_per_epoch=steps_per_epoch,
    #                       epochs=epochs_eva, gamma = gamma, polyak=0.995,
    #                       lr=lr, batch_size=batch_size, primary=True,
    #                       logger_kwargs=logger_kwargs)

    # # Freeze new policy but activate the updatability of the old policy (for evaluation only)
    # updatability(ac_new, ac_old)

    # Update the cost function with the same functional form but much higher precision
    cost_class = CostFunction(env_fn, act_partition=act_partition,
                              state_gen_runs=state_gen_runs_eva,
                              lc=lc, tc=tc)
    cost_fn = cost_class.duo

    # Decision of NAC
    nac_decision(ac_old, ac_new, o_initial, cost_fn, act_gen_runs=act_gen_runs_eva,
                 seed=seed, logger_kwargs=logger_kwargs)

    logger.log_tabular('ALgoTime', time.time()-timept_collect)

    # The end of NAC algorithm #

    # The following are online evaluations and testing on the decision if ideally we
    # want to check the nearly true net values of both policies

    # Update the net Q-networks of new policy via onlie evaluation
    updatability(ac_old, ac_new)


    online_eva(env_fn, ac_old, ac_new, o_initial, act_partition=act_partition,
               rbsize = int(1e6), batch_size = batch_size,
               state_gen_runs = state_gen_runs_eva, lc = lc, tc = tc, seed = seed,
               method = 'Two_Components', primary=True, update_every = 50,
               steps_per_epoch = steps_per_epoch, epochs = epochs_eva, gamma = gamma,
               lr = lr, polyak = 0.995, max_ep_len=max_ep_len, device = 'cpu',
               save_freq = save_freq, logger_kwargs=logger_kwargs)

    # Decision with (nearly) underlying truths
    nac_decision(ac_old, ac_new, o_initial, cost_fn, act_gen_runs=act_gen_runs_eva,
                seed=seed, logger_kwargs=logger_kwargs)
    
    logger.log_tabular('TotalTime', time.time()-start_time)
    logger.dump_tabular()