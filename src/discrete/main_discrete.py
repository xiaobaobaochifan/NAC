import copy
import torch
import argparse
import multiprocessing
from algs.run_solo_discrete import run


def wrapper(kwargs):
    '''Wrapper function to unpack arguments and call the target function.'''
    torch.set_num_threads(3)
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
                        help='Number of replications to run')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--env_name', type=str, default="sumo-rl-v0")
    parser.add_argument('--offline_size', type=int, default=int(1e5))
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--scratch', type=bool, default=False)
    parser.add_argument('--act_partition', type=str, default="default")
    parser.add_argument('--lc', type=int, default=10) # 50
    parser.add_argument('--tc', type=int, default=1) #1
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--state_gen_runs_trn', type=int, default=10)
    parser.add_argument('--state_gen_runs_eva', type=int, default=10000)
    parser.add_argument('--steps_per_epoch', type=int, default=400)
    parser.add_argument('--epochs_trn', type=int, default=150)
    parser.add_argument('--epochs_eva', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--start_steps', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--num_test_episodes', type=int, default=10)
    parser.add_argument('--max_ep_len', type=int, default=100)
    parser.add_argument('--q_grad_clip', type=float, default=1.0)
    parser.add_argument('--pi_grad_clip', type=float, default=1.0) # 3
    parser.add_argument('--rate_stop', type=float, default=0.5)
    parser.add_argument('--epochs_stop', type=int, default=20)
    parser.add_argument('--upper_stop', type=float, default=5.0)
    parser.add_argument('--lower_stop', type=float, default=5.0)
    parser.add_argument('--start_optimal', type=bool, default=False)
    args = parser.parse_args()

    main(args)

