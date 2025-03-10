import copy
import torch
import argparse
import multiprocessing
from algs.run_solo import run


def wrapper(kwargs):
    '''Wrapper function to unpack arguments and call the target function.'''
    torch.set_num_threads(2)
    run(**kwargs)

def main(args):
    args_dict = vars(args)

    # Remove 'reps' key to prevent it from being passed to compute_details
    reps = args_dict.pop('reps', 1)

    processes = []
    for i in range(reps):
        # Using deepcopy if kwargs_dict contains mutable objects that might be modified in processes
        process_kwargs = copy.deepcopy(args_dict)

        base_exp_name = process_kwargs.pop('exp_name', 'exps')  # Provide a default exp name
        name = f"{base_exp_name}_rep{i}"
        process_kwargs['logger_kwargs'] = dict(exp_name=name)
        process_kwargs['seed'] += i

        process = multiprocessing.Process(target=wrapper, args=(process_kwargs,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=5, required=True,
                        help='Number of replications to run')   # total number of replications
    parser.add_argument('--exp_name', type=str, required=True)  # name of the experiment
    parser.add_argument('--env_name', type=str, default="Ant-v4")   # name of the environment
    parser.add_argument('--offline_size', type=int, default=int(1e6))   # size of offline data
    parser.add_argument('--batch_size', type=int, default=256)  # batch size
    parser.add_argument('--scratch', type=bool, default=False)  # whether train a policy from scratch
    parser.add_argument('--act_partition', type=str, default="default") # choice of ways to partition the action space
    parser.add_argument('--lc', type=float, default=5)    # learning cost coefficient
    parser.add_argument('--tc', type=float, default=0.1)  # transaction cost coefficient
    parser.add_argument('--seed', type=int, default=0)  # starting value of consecutive random seeds
    parser.add_argument('--state_gen_runs_trn', type=int, default=10)   # number of sampled states in training to compute costs
    parser.add_argument('--state_gen_runs_eva', type=int, default=10000)    # number of samples states in evalaution to compute costs
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--epochs_trn', type=int, default=1000) # number of epochs in training
    parser.add_argument('--epochs_eva', type=int, default=100)  # number of epochs in evaluation
    parser.add_argument('--gamma', type=float, default=0.99)    # discounting factor
    parser.add_argument('--lr', type=float, default=0.0003) # learning rate
    parser.add_argument('--act_gen_runs_trn', type=int, default=1000)   # number of sampled actions in training policies
    parser.add_argument('--act_gen_runs_eva', type=int, default=10000)   # number of sampled actions in evaluating policies
    parser.add_argument('--save_freq', type=int, default=1) # frequency of saving intermediate results
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--max_ep_len', type=int, default=1000) # maximum length of one episode in sampling
    parser.add_argument('--q_grad_clip', type=float, default=1.0)
    parser.add_argument('--pi_grad_clip', type=float, default=1.0) # 3
    parser.add_argument('--rate_stop', type=float, default=1.0)
    parser.add_argument('--epochs_stop', type=int, default=20)
    parser.add_argument('--upper_stop', type=float, default=50.0)
    parser.add_argument('--lower_stop', type=float, default=10.0)
    parser.add_argument('--start_optimal', type=bool, default=False)    # whether to start training from an optimal policy
    args = parser.parse_args()

    main(args)

