import random
import numpy as np
import torch


from .optimization import get_custom_cosine_schedule_with_warmup, get_custom_linear_schedule_with_warmup

def set_seed(seed: int, device_type='cuda'):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, and `torch`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    # elif device_type == 'tpu':
    #     import torch_xla.core.xla_model as xm
    #     xm.set_rng_state(seed)
    torch.use_deterministic_algorithms(mode=True, warn_only=True) # Needed for reproducible results
    