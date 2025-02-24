from copy import deepcopy
import numpy as np
import torch
import time
from algs.nac_eva_discrete import nac_evaluate
from utils.logx import EpochLogger


def eva_fixed(ac_old, ac_new, o_initial, cost_fn, primary=False):

    # Ensure o has a batch dimension if it's a single state (feature dimension only)
    if o_initial.dim() == 1:
        o_initial = o_initial.unsqueeze(0)  # Add a batch dimension

    # Generate action probs from the current policy for the single state
    _, probs = ac_new.pi(o_initial, with_prob=True)

    # Cost
    if not primary:
        cost = cost_fn(ac_old, ac_new)
    else:
        cost = 0

    # net Q-values 
    nq1 = ac_new.q1(o_initial) - cost
    nq2 = ac_new.q2(o_initial) - cost

    # Conservative estimates
    nq_pi = torch.min(nq1, nq2)
    nq_pi = torch.sum(nq_pi * probs)

    # Useful info for logging
    # pi_info = dict(LogPi=logp_pi.detach().numpy())

    # return loss_pi, pi_info
    return nq_pi

# Freeze all other parameters and only reactivate nq parameters of ac_old
def updatability(ac_new, ac_old):
    for param in ac_new.parameters():
        param.requires_grad = False

    for param in ac_old.parameters():
        param.requires_grad = True

def eva_old(env_fn, offline_data, ac_old, cost_fn,  
            seed = 0, steps_per_epoch=1000, epochs=100, gamma=0.99, 
            polyak=0.995, lr=3e-4, batch_size=256, primary=True,
            logger_kwargs=dict()):
    
    ac_old_ref = deepcopy(ac_old)

    ac_old = nac_evaluate(env_fn, offline_data, ac_old_ref, ac_old, cost_fn,  
              seed, steps_per_epoch, epochs, gamma, 
              polyak, lr, batch_size, primary, logger_kwargs)
    
    return ac_old

# Only make a decision if satisfied with nq-networks of both old and new policies
def nac_decision(ac_old, ac_new, o_initial, cost_fn, seed=0, logger_kwargs=dict()):
    start_time = time.time()

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    eva_old = eva_fixed(ac_old, ac_old, o_initial, cost_fn, primary = True)
    eva_new = eva_fixed(ac_old, ac_new, o_initial, cost_fn, primary = False)

    if eva_old >= eva_new:
        decision = 'Old'
    else:
        decision = 'New'

    logger.log_tabular('Eva_old', eva_old.item())
    logger.log_tabular('Eva_new', eva_new.item())
    logger.log_tabular('Decision', decision)
    logger.log_tabular('Time', time.time()-start_time)
    logger.dump_tabular()

    return [decision, [[ac_old, eva_old], [ac_new, eva_new]]]
