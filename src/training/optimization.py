import math
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def _get_custom_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, 
    *, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    num_cycles: float,
    min_lr_ratio: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    lr_factor = 0.5 * (1.0 + math.cos(2.0 * math.pi * float(num_cycles) * progress))  # cosine decay
    lr_factor = min_lr_ratio + (1.0 - min_lr_ratio) * lr_factor  # rescaling according to min_lr_ratio
    return max(min_lr_ratio, lr_factor)


def get_custom_cosine_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    min_lr_ratio: float,
    num_cycles: float = 0.5, 
    last_epoch: int = -1,
):
    lr_lambda = partial(
        _get_custom_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_custom_linear_schedule_with_warmup_lr_lambda(
    current_step: int, 
    *, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    num_cycles: float,
    min_lr_ratio: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    lr_factor = 1.0 - progress # linear decay
    lr_factor = min_lr_ratio + (1.0 - min_lr_ratio) * lr_factor  # rescaling according to min_lr_ratio
    return max(min_lr_ratio, lr_factor)

def get_custom_linear_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    min_lr_ratio: float,
    num_cycles: float = 0.5, 
    last_epoch: int = -1,
):
    lr_lambda = partial(
        _get_custom_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)