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
import sumo_rl


def sac(env_name, actor_critic=core.DiscreteMLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=1000, epochs=1000, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=256, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
 

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env_fn = lambda: gym.make('sumo-rl-v0',
                net_file='utils/nets/single-intersection/single-intersection.net.xml',
                route_file='utils/nets/single-intersection/single-intersection.rou.xml',
                out_csv_name='output/csvs/single-intersection/output.csv',
                use_gui=False,
                num_seconds=6000000, single_agent=True, reward_fn='average-speed')
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    # Create actor-critic module and target networks
    obs_space = Box(low=env.observation_space.low, high=env.observation_space.high,
                        shape=env.observation_space.shape, dtype=np.float32)
    act_space = Discrete(n=env.action_space.n)
    ac = actor_critic(obs_space, act_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # # Count variables (protip: try to get a feel for how different size networks behave!)
    # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):

        o, a, r, o2, d, _ = data['obs'], data['act'], data['rew'], data['obs2'], data['trm'], data['trc']

        # Ensure actions are long tensors
        a = a.long()

        q1 = ac.q1(o).gather(1, a.unsqueeze(-1)).squeeze(-1)
        q2 = ac.q2(o).gather(1, a.unsqueeze(-1)).squeeze(-1)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, prob = ac.pi(o2, with_prob=True)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2).gather(1, a2.unsqueeze(-1)).squeeze(-1)
            q2_pi_targ = ac_targ.q2(o2).gather(1, a2.unsqueeze(-1)).squeeze(-1)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            # Compute the entropy term
            entropy_term = - alpha * torch.sum(prob * torch.log(prob + 1e-8), dim=1)
        
            # Bellman backup with entropy regularization
            backup = r + gamma * (1 - d) * (q_pi_targ + entropy_term)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):

        o = data['obs']
        pi, prob = ac.pi(o, with_prob=True)
        pi = pi.long()  # Ensure sampled actions are long

        q1_pi = ac.q1(o).gather(1, pi.unsqueeze(-1)).squeeze(-1)
        q2_pi = ac.q2(o).gather(1, pi.unsqueeze(-1)).squeeze(-1)
        q_pi = torch.min(q1_pi, q2_pi)

        # Compute the entropy term
        entropy_term = - torch.sum(prob * torch.log(prob + 1e-8), dim=1)
    
        # Entropy-regularized policy loss
        loss_pi = (- alpha * entropy_term - q_pi).mean()

        # Useful info for logging
        pi_info = dict(EntropyPi=entropy_term.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, info, trm, trc, ep_ret, ep_len = *test_env.reset(), False, False, 0, 0
            while not(trm or trc or (ep_len == max_ep_len)):
                # Take non-deterministic actions at test time 
                o, r, trm, trc, _ = test_env.step(get_action(o, False))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, _, ep_ret, ep_len = *env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, trm, trc, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, trm, trc)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if trm or trc or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, _, ep_ret, ep_len = *env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('EntropyPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()



