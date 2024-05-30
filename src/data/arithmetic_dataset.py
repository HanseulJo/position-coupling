import os
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotmap import DotMap
from omegaconf import OmegaConf

from src.tokenization import tokenize_for_decoder, tokenize_for_encoder_decoder
from src.model import DECODER_BASED, ENCODER_DECODER_BASED
from src.data.common import OPERATORS, Operation

# The actual implementation of datasets are here!
from src.data.addition import *
from src.data.multiplication import *
from src.data.copy import *
from src.data.parity import *


## Build Dataset (train / val / val_long)
def build_dataset(cfg, verbose=True):
    task_cfg = OmegaConf.to_container(cfg.task)
    symbol = task_cfg['symbol']
    n_input = OPERATORS[symbol]['n_input']
    operation = OPERATORS[symbol]['operation']

    operation = Operation(symbol, n_input, operation)
    dataset = {}
    phases = ['train', 'val', 'val_hard', 'val_long', 'val_long_hard']
    common_task_cfg = task_cfg.copy()
    for k in phases:
        if k in common_task_cfg:
            common_task_cfg.pop(k)
    if verbose: print("Preparing Dataset...")
    for phase in phases:
        if phase not in task_cfg:
            continue
        if verbose: print(f"{phase}...")
        dataset_cls = eval(task_cfg[phase].pop('dataset_cls'))
        dataset[phase] = dataset_cls(
            operation=operation, 
            **task_cfg[phase],
            **common_task_cfg
        )
    if verbose: print()
    
    ## Store datasets ##
    data_path = f"./dataset/seed{cfg.seed_data}"
    if not os.path.exists(data_path): 
        os.makedirs(data_path)
    for phase in dataset.keys():
        pbar = tqdm(dataset[phase], disable=not verbose)
        with open(f"{data_path}/{phase}_input.txt", "w") as f_i, \
            open(f"{data_path}/{phase}_label.txt", "w") as f_o:
            for i, item in enumerate(pbar):
                if i >= 100: break
                input, label = item[:2]
                f_i.write(input+"\n")
                f_o.write(label+"\n")
    return dataset


def tokenize_fn(x, cfg, tokenizer, device, arr_type):
    if cfg.model.model_name in ENCODER_DECODER_BASED:
        fn = tokenize_for_encoder_decoder
    elif cfg.model.model_name in DECODER_BASED:
        fn = tokenize_for_decoder
    else:
        raise ValueError(f"model_name: {cfg.model.model_name}")
    return fn(tokenizer, *zip(*x), device=device, arr_type=arr_type)
    

def build_loader(
        cfg, 
        dataset, 
        tokenizer, 
        device='cpu', 
        sampler=None, 
        num_workers=0, 
        arr_type='torch'
    ):
    if sampler is None:
        sampler = {}
        for phase in dataset:
            sampler[phase] = None
    
    loader = {}
    for phase in dataset:
        loader[phase] = DataLoader(
            dataset[phase], 
            batch_size=(
                cfg.training.batch_size_train 
                if phase == 'train' 
                else cfg.training.batch_size_eval
            ), 
            shuffle=(phase == 'train') and sampler[phase] is None,
            sampler=sampler[phase],
            collate_fn=partial(
                tokenize_fn, 
                cfg=cfg, 
                tokenizer=tokenizer, 
                arr_type=arr_type,
                device=device),
            num_workers=num_workers,)
    
    return loader


def build_auxiliary_dataset(cfg):
    task_cfg = DotMap(OmegaConf.to_container(cfg.task))
    symbol = task_cfg.symbol
    n_input = OPERATORS[symbol]['n_input']
    operation = OPERATORS[symbol]['operation']

    operation = Operation(symbol, n_input, operation)
    auxiliary_dataset = eval(task_cfg.auxiliary.pop('dataset_cls'))(
        operation, 
        reverse_output=task_cfg.reverse_output, 
        commutative=task_cfg.commutative,
        **task_cfg.train.toDict(), # same setup as train data
        **task_cfg.auxiliary.toDict()
    )
    
    ## Store datasets ##
    data_path = f"./dataset/seed{cfg.seed}"
    if not os.path.exists(data_path): 
        os.makedirs(data_path)
    
    pbar = tqdm(auxiliary_dataset)
    with open(f"{data_path}/train_aux_input.txt", "w") as f_i, \
        open(f"{data_path}/train_aux_label.txt", "w") as f_o:
        for item in pbar:
            input, label = item[:2]
            f_i.write(input+"\n")
            f_o.write(label+"\n")
    
    return auxiliary_dataset


def build_auxiliary_loader(cfg, auxiliary_dataset, tokenizer):
    if cfg.model.model_name in ENCODER_DECODER_BASED:
        tokenize = tokenize_for_encoder_decoder
    elif cfg.model.model_name in DECODER_BASED:
        tokenize = tokenize_for_decoder
    else:
        raise ValueError(f"model_name: {cfg.model.model_name}")

    loader = DataLoader(auxiliary_dataset, batch_size=cfg.training.batch_size_train, shuffle=True, 
                        collate_fn=lambda x: tokenize(tokenizer, *zip(*x), device=cfg.device))
    return loader