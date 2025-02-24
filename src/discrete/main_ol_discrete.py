import copy
import torch
import argparse
import multiprocessing
from algs.sac_discrete import sac


def wrapper(kwargs):
    '''Wrapper function to unpack arguments and call the target function.'''
    torch.set_num_threads(10)
    sac(**kwargs)

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
                        help='Number of replications to run')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--env_name', type=str, default="sumo-rl-v0")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps_per_epoch', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--replay_size', type=int, default=int(1e5))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--start_steps', type=int, default=1000)
    parser.add_argument('--update_after', type=int, default=100)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--max_ep_len', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=1)
    args = parser.parse_args()

    main(args)

