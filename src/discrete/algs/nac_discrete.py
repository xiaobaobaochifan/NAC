'''
This is the main module to conduct NAC policy learning, with state space 
to be discrete. We first use offline data to learn the best candidate policy by nac_learn,
then fine-tune the evaluation of the output candidate policy with nac_evaluation, finally we
decide whether to switch to that policy with nac_decision 
'''

from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import time
import algs.core_discrete as core
from algs.replay_buffer_discrete import ReplayBuffer
from utils.logx import EpochLogger

# Offline data collection
# Different from SAC code, we need another function named 
# "offline_collection" which first stores transition tuples
# according to uniform random policy and then according to
# the given old policy until it reaches the size limit 
# (offline_size), by updating the instance offline_data
def offline_collection(offline_data, env_fn, ac_old, size, start_steps, max_ep_len=1000):
    
    env = env_fn()
    
    def get_action_old(obs, ac_old=ac_old, deterministic=False):
        # The function to sample an action according to the old policy in 'actor_critic_old'
        return ac_old.act(torch.as_tensor(obs, dtype=torch.float32), deterministic)

    o, info = env.reset()
    ep_len = 0

    for t in range(size):

        # Start from the uniform random policiy for the first 'start_steps' number of steps.
        # The rest data would be generated due to the old policy from 'actor_critic_old'.
        if t <= start_steps:
            a = env.action_space.sample()
        else:
            a = get_action_old(o)

        # One new step of transition tuple
        o2, r, trm, trc, _ = env.step(a)
        ep_len += 1

        # Store experience to offline data pool
        offline_data.store(o, a, r, o2, trm, trc)

        # Update the current state
        o = o2

        # End of trajectory handling
        if trm or trc or (ep_len == max_ep_len):
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, _ = env.reset()
            ep_len = 0

        # print(f'Step: {t}')


def offline_generation(env_fn, ac_old, offline_size=int(1e6), start_steps=10000, max_ep_len=1000):

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n
    
    # Create an initial offline buffer 
    # (to be updated datum by datum by interaction with environments)
    offline_data = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=offline_size)

    # Fill replay buffer with actual offline data
    offline_collection(offline_data, env_fn, ac_old, offline_size, start_steps, max_ep_len)

    return offline_data

def state_init_generator(env_name, env_fn):
    env = env_fn()
    obs_space = Box(low=env.observation_space.low, high=env.observation_space.high,
                    shape=env.observation_space.shape, dtype=np.float32)
    obs_dim = obs_space.shape
    o_initial = torch.zeros(obs_dim, dtype=torch.float32)
    if env_name == 'sumo-rl-v0':
        o_initial[0] = 1.0
        
    return o_initial

def nac_learn(env_fn, ac_old, cost_fn, o_initial, offline_data, old_ref,
              actor_critic=core.DiscreteMLPActorCritic, 
              ac_kwargs=dict(hidden_sizes=[256]*2), seed=0, steps_per_epoch=1000, epochs=1000,
              gamma=0.99, polyak=0.995, lr=3e-4, batch_size=256, 
              logger_kwargs=dict(), save_freq=1, scratch=False, num_test_episodes=10,
              max_ep_len=1000, q_grad_clip=1.0, pi_grad_clip=1.0, rate_stop=1.0,
              epochs_stop=22, upper_stop=20, lower_stop=10):
    
    '''
    Net Actor Critic (NAC), learning part:
    The main function to learn switch-optimal policy, should be followed by finer 
    evaluation with the function nac_evaluation and finally the decision with nac_decision

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
    if scratch == True:
        obs_space = Box(low=env.observation_space.low, high=env.observation_space.high,
                        shape=env.observation_space.shape, dtype=np.float32)
        act_space = Discrete(n=env.action_space.n)
        ac_new = actor_critic(obs_space, act_space, **ac_kwargs)
    else:
        ac_new = deepcopy(ac_old)

    ac_targ = deepcopy(ac_new)

    # Freeze target networks with respect to optimizers 
    # (only allow manual update, i.e., polyak averaging)
    for param in ac_targ.parameters():
        param.requires_grad = False

    for param in ac_old.parameters():
        param.requires_grad = False

    # Save all initial parameters of all Q-networks of
    # the new policy (candidate, to be updated)
    q_params = list(itertools.chain(ac_new.q1.parameters(), ac_new.q2.parameters()))


    # Function to compute NAC net Q-losses
    def compute_loss_nq(data, cost_fn, ac_old, ac_new, ac_targ, gamma):
        o, a, r, o2, d, _ = data['obs'], data['act'], data['rew'], data['obs2'], data['trm'], data['trc']

        # Ensure actions are long tensors
        a = a.long()
        
        # net Q-values for the taken actions
        nq1 = ac_new.q1(o).gather(1, a.unsqueeze(-1)).squeeze(-1)
        nq2 = ac_new.q2(o).gather(1, a.unsqueeze(-1)).squeeze(-1)

        # Net Bellman backup for net Q-functions
        with torch.no_grad():
            # Target actions from the current candidate policy
            a2, _ = ac_new.pi(o2)
            
            # Target net Q-values for the selected actions
            nq1_targ = ac_targ.q1(o2).gather(1, a2.unsqueeze(-1)).squeeze(-1)
            nq2_targ = ac_targ.q2(o2).gather(1, a2.unsqueeze(-1)).squeeze(-1)
            nq_targ = torch.min(nq1_targ, nq2_targ)
            backup = r + gamma * (1 - d) * (nq_targ)

        # MSE loss against Bellman backup
        loss_nq1 = ((nq1 - backup)**2).mean()
        loss_nq2 = ((nq2 - backup)**2).mean()
        loss_nq = loss_nq1 + loss_nq2

        # Useful info for logging
        # q_info = dict(Q1Vals=q1.detach().numpy(),
        #                 Q2Vals=q2.detach().numpy())

        # return loss_q, q_info
        return loss_nq

    # Function to compute pi_loss
    def compute_loss_pi(o_initial, cost_fn, ac_old, ac_new):
        
        # Ensure o_initial is of shape (1, obs_dim) for compatibility
        o_initial = o_initial.unsqueeze(0)

        # Generate action probs from the current policy for the single state
        _, probs = ac_new.pi(o_initial, with_prob=True)

        # Compute cost
        cost = cost_fn(ac_old, ac_new)        
        
        # Compute Q-values for all actions in the initial state
        nq1 = ac_new.q1(o_initial) - cost  # Shape: [1, act_dim]
        nq2 = ac_new.q2(o_initial) - cost  # Shape: [1, act_dim]

        # Conservative estimates: take the minimum Q-value for each action
        nq_pi = torch.min(nq1, nq2)  # Shape: [1, act_dim]

        # Compute the policy loss: expectation of q_pi with respect to action probabilities
        # Element-wise multiplication of q_pi and probs followed by summation across actions
        loss_pi = -torch.sum(nq_pi * probs)

        # Useful info for logging
        # pi_info = dict(LogPi=logp_pi.detach().numpy())

        # Return the loss
        return loss_pi

    # Set up optimizers for policy and net Q-function
    pi_optimizer = Adam(ac_new.pi.parameters(), lr=lr)
    nq_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac_new)

    # The function for updates
    def update(data, cost_fn, ac_old, ac_new, ac_targ):
        # One step update on parameters of both net Q-functions and policy pi of ac_new

        # Insert this into the main flow before calculating the gradient of net Q-losses
        # to ensure, when updating net Q parameters, the component of cost function, containg
        # the parameters of current candidate new actor-critic, will not affect the computational
        # graph
        for param in ac_new.pi.parameters():
            param.requires_grad = False

        # One step of gradient descent update for all net Q-functions
        nq_optimizer.zero_grad()
        loss_nq = compute_loss_nq(data, cost_fn, ac_old, ac_new, ac_targ, gamma)
        loss_nq.backward()

        # # Calculate and print the norm of gradients for Q-networks
        # q_grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in q_params if p.grad is not None))
        # print(f"Norm of gradients for Q-networks before clipping: {q_grad_norm}")

        # Clip gradients for the Q-networks
        # torch.nn.utils.clip_grad_norm_(q_params, max_norm=q_grad_clip)

        nq_optimizer.step()

        # Unfreeze pi-network for the folloing policy learning step
        for param in ac_new.pi.parameters():
            param.requires_grad = True

        # Record things
        logger.store(LossNQ=loss_nq.item())

        # Freeze Q-networks 
        for param in q_params:
            param.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(o_initial, cost_fn, ac_old, ac_new)
        loss_pi.backward()

        # # Calculate and print the norm of gradients for the policy network
        # pi_grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in ac_new.pi.parameters() if p.grad is not None))
        # print(f"Norm of gradients for policy network before clipping: {pi_grad_norm}")

        # Clip gradients for the policy network
        # torch.nn.utils.clip_grad_norm_(ac_new.pi.parameters(), max_norm=pi_grad_clip)

        pi_optimizer.step()

        # Unfreeze Q-networks for the next iteration.
        for param in q_params:
            param.requires_grad = True

        # Record things
        logger.store(LossPi=-loss_pi.item())

        # Update target Q-networks only through polyak averaging.
        with torch.no_grad():
            for param, param_targ in zip(ac_new.parameters(), ac_targ.parameters()):
                # Use inplace operations to avoid creating new tensors
                param_targ.mul_(polyak)
                param_targ.add_((1 - polyak) * param)

        return loss_pi, loss_nq

    def get_action(o, deterministic=False):
        return ac_new.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def test_agent(test_env, num_test_episodes, max_ep_len):
        for _ in range(num_test_episodes):
            o, info, trm, trc, ep_ret, ep_len = *test_env.reset(), False, False, 0, 0
            while not(trm or trc or (ep_len == max_ep_len)):
                # Take non-deterministic actions at test time 
                o, r, trm, trc, stepinfo = test_env.step(get_action(o, False))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for training
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    # o_initial = torch.zeros(obs_dim, dtype=torch.float32)

    stop_flag = False  # Initialize the stop flag as False
    recent_pi_losses = []  # List to keep track of the last three losses

    # Main flow: update per step and log per epoch
    for t in range(total_steps):

        if stop_flag:  # Check if stop condition was met in the previous epoch
            break  # Exit the loop early

        # Update parameters once in each step, with a total of total_steps
        batch = offline_data.sample_batch(batch_size)
        loss_pi, loss_nq = update(batch, cost_fn, ac_old, ac_new, ac_targ)

        nq_optimizer.zero_grad()
        pi_optimizer.zero_grad()

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # print(-loss_pi.item(), old_ref)

            # Update the recent losses list
            recent_pi_losses.append(-loss_pi.item())
            if len(recent_pi_losses) > 2:  # Keep only the last three entries
                recent_pi_losses.pop(0)

            # Check if early stopping conditions are met
             # Set the stop flag to True to stop training in the next iteration
            if (len(recent_pi_losses) == 2) and epoch >= epochs_stop:
                if old_ref > 0:
                    if all(loss > (1+rate_stop)*old_ref for loss in recent_pi_losses):
                        stop_flag = True
                else:
                    if all(loss > 0 for loss in recent_pi_losses):
                        stop_flag = True
                if all(loss >= old_ref+upper_stop for loss in recent_pi_losses):
                    stop_flag = True
                elif all(loss <= old_ref-lower_stop for loss in recent_pi_losses):
                    stop_flag = True

             # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                    logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent(test_env, num_test_episodes, max_ep_len)

            # Log info about epoch
            logger.log_tabular('Epoch_training', epoch)
            logger.log_tabular('TotalSteps', t+1)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossNQ', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
    
    return ac_new
