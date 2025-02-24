import torch
from copy import deepcopy
from torch.optim import Adam
import itertools
import numpy as np
import time
from algs.costs import CostFunction
from algs.replay_buffer import ReplayBuffer
from utils.logx import EpochLogger


'''
Online net value evaluation on the learned policy pi_n
''' 
def online_eva(env_fn, ac_old, ac_new, o_initial, act_partition, rbsize = int(1e6), 
                             batch_size = 100, state_gen_runs = int(1e5), lc = 5, tc = 0, 
                             seed = 0, method = 'Two_Components', primary=False,
                             update_every = 50, steps_per_epoch = 1000, epochs = 10, 
                             gamma = 0.99, lr = 1e-3, polyak = 0.995, max_ep_len=1000,
                             device = 'cpu', logger_kwargs=dict(), save_freq=1):
    

    # # This will construct empty logger dictionary, i.e. logger_current_row = {}, and save all ouput puts to output_dir/progress.txt (if no filename is specifed).
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    online_data = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=rbsize)

    ac_targ = deepcopy(ac_new)

    # Freeze all parameters in the target network
    for p in ac_targ.parameters():
        p.requires_grad = False
    # Freeze all the parameters in the old actor ctiric
    for p in ac_old.parameters():
        p.requires_grad = False
    # Freeze all parameters in policy networks of the new actor critic
    for param in ac_new.pi.parameters():
        param.requires_grad = False

    q_params = itertools.chain(ac_new.q1.parameters(), ac_new.q2.parameters())
    nq_optimizer = Adam(q_params, lr=lr)

    # var_counts = tuple(actor_critic.count_vars(module) for module in [ac_new.q1, ac_new.q2])
    # logger.log('\nNumber of parameters: \t q1: %d, \t q2: %d\n'%var_counts)

    cost_class = CostFunction(env_fn, act_partition, state_gen_runs, lc, tc, device)
    if not primary:
        if method == 'Two_Components':
            cost = cost_class.duo(ac_old, ac_new)
        # if method == 'Wasserstein_L2':
        #     cost = cost = cost_function.WassersteinL2(ac_old.pi, ac_new.pi)
    else:
        cost = 0

    # Setup the model that need to be saved at the end of each epoch. 
    logger.setup_pytorch_saver(ac_new)

    # Function to compute NAC net Q-losses
    def compute_loss_nq(data, cost, ac_new, ac_targ):
        o, a, r, o2, d, _ = data['obs'], data['act'], data['rew'], data['obs2'], data['trm'], data['trc']

        # net Q-values
        nq1 = ac_new.q1(o,a) - cost
        nq2 = ac_new.q2(o,a) - cost

        # Net Bellman backup for net Q-functions
        with torch.no_grad():
            # Target actions from the current candidate policy
            a2, _, _ = ac_new.pi(o2)

            # Target net Q-values
            cost_targ = cost
            nq1_targ = ac_targ.q1(o2, a2) - cost_targ
            nq2_targ = ac_targ.q2(o2, a2) - cost_targ
            nq_targ = torch.min(nq1_targ, nq2_targ)
            backup = r + gamma * (1 - d) * (nq_targ) - (1 - gamma) * cost
            # check later if the cost here should be from targ or current new policy

        # MSE loss against Bellman backup
        loss_nq1 = ((nq1 - backup)**2).mean()
        loss_nq2 = ((nq2 - backup)**2).mean()
        loss_nq = loss_nq1 + loss_nq2

        # Useful info for logging
        # q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
        #                 Q2Vals=q2.cpu().detach().numpy())

        # return loss_q, q_info
        return loss_nq

    # Define update function only for Q-function.
    def update(data, cost, ac_new, ac_targ):

        # Update the parameters of ac_n.q1 and ac_n.q2
        nq_optimizer.zero_grad()
        loss_nq = compute_loss_nq(data, cost, ac_new, ac_targ)
        loss_nq.backward()
        nq_optimizer.step() 

        # Update the parameters of ac_n_targ
        with torch.no_grad():
            for param, param_targ in zip(ac_new.parameters(), ac_targ.parameters()):
                # Use inplace operations to avoid creating new tensors
                param_targ.mul_(polyak)
                param_targ.add_((1 - polyak) * param)

        # Log loss values of net Q
        logger.store(LossNQ = loss_nq.item())

    def get_action(obs, deterministic=False):
        return ac_new.act(torch.as_tensor(obs, dtype=torch.float32), 
                      deterministic)

    # Main loop begins here.
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, info = env.reset()
    ep_len = 0

    for t in range(total_steps):

        a = get_action(o)
        o2, r, trm, trc, _ = env.step(a)
        ep_len += 1
        
        # Both observations and actions in the replay buffer are numpy array
        online_data.store(o, a, r, o2, trm, trc)
        o = o2

        # End of trajectory handling
        if trm or trc or (ep_len == max_ep_len):
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, _ = env.reset()
            ep_len = 0

        # Update
        if t % update_every == 0:
            for j in range(update_every):
                batch = online_data.sample_batch(batch_size)
                update(batch, cost, ac_new, ac_targ)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save the model
            if  (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            logger.log_tabular('Epoch_eva_on', epoch)
            logger.log_tabular('TotalInteracts', t+1)
            logger.log_tabular('LossNQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


    return ac_new


        