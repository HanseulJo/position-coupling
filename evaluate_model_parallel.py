import argparse
import copy
from itertools import product
import multiprocessing as mp
import evaluate_model
import evaluate_model_multiple_addition
import evaluate_model_multiplication
import evaluate_model_heatmap
import time
import os 
os.environ['PJRT_DEVICE'] = 'GPU'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',    type=str,   default='./configs')
    parser.add_argument('--config_name',    type=str,   default='config')
    parser.add_argument('--runner_name',    type=str,   default='evaluate_model')
    parser.add_argument('--group_name', type=str, default='test')
    parser.add_argument('--exp_name',   type=str, default='test')
    parser.add_argument('--min_n_digits',type=int,  default=1)
    parser.add_argument('--max_n_digits',type=int,  default=100)
    parser.add_argument('--min_n_operands',  type=int,  default=2)
    parser.add_argument('--max_n_operands',  type=int,  default=30)
    parser.add_argument('--min_n_digits_1',type=int,  default=1)
    parser.add_argument('--max_n_digits_1',type=int,  default=30)
    parser.add_argument('--min_n_digits_2',  type=int,  default=1)
    parser.add_argument('--max_n_digits_2',  type=int,  default=30)
    parser.add_argument('--step_digits',     type=int,  default=1)
    parser.add_argument('--step_operands',   type=int,  default=1)
    parser.add_argument('--step_digits_1',     type=int,  default=1)
    parser.add_argument('--step_digits_2',   type=int,  default=1)
    parser.add_argument('--compile',   action='store_true')
    parser.add_argument('--seeds',      type=int, default=[0],  nargs='*')
    parser.add_argument('--seeds_data',      type=int, default=[0],  nargs='*')
    parser.add_argument('--devices',    type=int, default=[0],  nargs='*')
    parser.add_argument('--num_exp_per_device', type=int,   default=1)
    parser.add_argument('--overrides',  type=str, default=[],   nargs='*')
    args = vars(parser.parse_args())

    runner = eval(args.pop('runner_name')).evaluate

    seeds = args.pop('seeds')
    seeds_data = args.pop('seeds_data')
    available_gpus = args.pop('devices')
    num_exp_per_device = args.pop('num_exp_per_device')

    experiments = []
    for seed, seed_data in product(seeds, seeds_data):
        exp = copy.deepcopy(args)
        group_name = exp.pop('group_name')
        exp_name = exp.pop('exp_name')
        exp['overrides'].append(f'group_name={group_name}')
        exp['overrides'].append(f'exp_name={exp_name}')
        exp['overrides'].append(f'seed={seed}')
        exp['overrides'].append(f'seed_data={seed_data}')
        experiments.append(exp)
    
    print(experiments)
    
    # run parallell experiments
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    mp.set_start_method('spawn')
    process_dict = {gpu_id: [] for gpu_id in available_gpus}
    
    for exp in experiments:
        wait = True

        # wait until there exists a finished process
        while wait:
            # Find all finished processes and register available GPU
            for gpu_id, processes in process_dict.items():
                for process in processes:
                    if not process.is_alive():
                        print(f"Process {process.pid} on GPU {gpu_id} finished.")
                        processes.remove(process)
                        if gpu_id not in available_gpus:
                            available_gpus.append(gpu_id)
            
            for gpu_id, processes in process_dict.items():
                if len(processes) < num_exp_per_device:
                    wait = False
                    gpu_id, processes = min(process_dict.items(), key=lambda x: len(x[1]))
                    break
            
            time.sleep(1)

        # get running processes in the gpu
        processes = process_dict[gpu_id]
        exp['overrides'].append(f'device=cuda:{gpu_id}')
        process = mp.Process(target=runner, args=(exp,))
        process.start()
        processes.append(process)
        print(f"Process {process.pid} on GPU {gpu_id} started.")

        # check if the GPU has reached its maximum number of processes
        if len(processes) == num_exp_per_device:
            available_gpus.remove(gpu_id)

