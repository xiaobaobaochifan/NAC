from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import time
from utils.logx import EpochLogger

def nac_evaluate(env_fn, offline_data, ac_old, ac_new, cost_fn,  
              seed = 0, steps_per_epoch=1000, epochs=100, gamma=0.99, 
              polyak=0.995, lr=3e-4, batch_size=256, primary=False,
              logger_kwargs=dict(), save_freq=10):
    
    '''
    Net Actor Critic (NAC), evaluation part:
    The main function to evaulate the learned switch-optimal policy or the old policy,
    should be followed by the decision with nac_decision

    
    Arguments:

        ac:

    '''

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set environments
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    # Initiate an actor-critic agent and a set of target networks
    ac_targ = deepcopy(ac_new)

    # Freeze target networks with respect to optimizers 
    # (only allow manual update, i.e., polyak averaging)
    for param in ac_targ.parameters():
        param.requires_grad = False

    for param in ac_old.parameters():
        param.requires_grad = False

    for param in ac_new.pi.parameters():
        param.requires_grad = False

    # Save all initial parameters of all Q-networks of
    # the given policy (to be updated)
    q_params = itertools.chain(ac_new.q1.parameters(), ac_new.q2.parameters())

    if not primary:
        cost = cost_fn(ac_old, ac_new)
    else:
        cost = 0

    # Function to compute NAC net Q-losses
    def compute_loss_nq(data, cost_fn, cost, ac_old, ac_new, ac_targ, primary):
        o, a, r, o2, d, _ = data['obs'], data['act'], data['rew'], data['obs2'], data['trm'], data['trc']

        # Ensure actions are long tensors
        a = a.long()
        
        # net Q-values
        nq1 = ac_new.q1(o).gather(1, a.unsqueeze(-1)).squeeze(-1) - cost
        nq2 = ac_new.q2(o).gather(1, a.unsqueeze(-1)).squeeze(-1) - cost

        # Net Bellman backup for net Q-functions
        with torch.no_grad():
            # Target actions from the current candidate policy
            a2, _ = ac_new.pi(o2)

            # Target net Q-values
            if not primary:
                cost_targ = cost_fn(ac_old, ac_targ)
            else:
                cost_targ = 0

            nq1_targ = ac_targ.q1(o2).gather(1, a2.unsqueeze(-1)).squeeze(-1) - cost_targ
            nq2_targ = ac_targ.q2(o2).gather(1, a2.unsqueeze(-1)).squeeze(-1) - cost_targ
            nq_targ = torch.min(nq1_targ, nq2_targ)
            backup = r + gamma * (1 - d) * (nq_targ) - (1 - gamma) * cost
            # check later if the cost here should be from targ or current new policy

        # MSE loss against Bellman backup
        loss_nq1 = ((nq1 - backup)**2).mean()
        loss_nq2 = ((nq2 - backup)**2).mean()
        loss_nq = loss_nq1 + loss_nq2

        # Useful info for logging
        # q_info = dict(Q1Vals=q1.detach().numpy(),
        #                 Q2Vals=q2.detach().numpy())

        # return loss_q, q_info
        return loss_nq
    
    # Set up optimizers for policy and net Q-function
    nq_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac_new)

    # The function for updates
    def update(data, cost_fn, cost, ac_old, ac_new, ac_targ):

        # One step of gradient descent update for all net Q-functions of ac_new
        nq_optimizer.zero_grad()
        loss_nq = compute_loss_nq(data, cost_fn, cost, ac_old, ac_new, ac_targ, primary)
        loss_nq.backward()
        nq_optimizer.step()

        # Record things
        logger.store(LossNQ=loss_nq.item())

        # Update target Q-networks only through polyak averaging.
        with torch.no_grad():
            for param, param_targ in zip(ac_new.parameters(), ac_targ.parameters()):
                # Use inplace operations to avoid creating new tensors
                param_targ.mul_(polyak)
                param_targ.add_((1 - polyak) * param)

    # Prepare for training
    total_steps = steps_per_epoch * epochs
    start_time = time.time()

    # Main flow: update per step and log per epoch
    for t in range(total_steps):
        # Update parameters once in each step, with a total of total_steps

        batch = offline_data.sample_batch(batch_size)
        update(batch, cost_fn, cost, ac_old, ac_new, ac_targ)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

             # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                    logger.save_state({'env': env}, None)

            # Log info about epoch
            logger.log_tabular('Epoch_eva_off', epoch)
            logger.log_tabular('TotalSteps', t+1)
            logger.log_tabular('LossNQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

    return ac_new