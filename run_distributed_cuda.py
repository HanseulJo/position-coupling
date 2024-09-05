import argparse
from contextlib import nullcontext
from dotmap import DotMap
from hydra import compose, initialize
import json
import math
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import time
import torch
import torchinfo
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_scheduler
import wandb
from warnings import filterwarnings
filterwarnings("ignore")

from src.tokenization import build_tokenizer
from src.data import build_dataset, build_loader
from src.model import build_model_from_scratch
from src.training import set_seed, get_custom_cosine_schedule_with_warmup, get_custom_linear_schedule_with_warmup
from src.evaluate import get_tokenwise_accuracy, get_instancewise_accuracy
from src.common import now, print_training_update

def ddp_setup(backend, rank, world_size):
    print(f"DDP Setting up... ({rank})")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "23457"
    init_process_group(
        backend=backend, 
        rank=rank,
        world_size=world_size
    )
    print(f"DDP Set up!: ({rank})")


def mp_fn(rank, cfg, device_ids, device_type, logging_path, use_wandb, wandb_run):
    backend = 'nccl'
    world_size = len(device_ids)
    ddp_setup(backend, rank, world_size)
    device_id = int(device_ids[rank])
    main(cfg, device_id, device_ids, device_type, logging_path, use_wandb, wandb_run)
    destroy_process_group()


def main(cfg, device_id, device_ids, device_type, logging_path, use_wandb, wandb_run):
    # Device
    device = torch.device(f'cuda:{device_id}')
    
    # Main process?
    in_main_process = device_id == device_ids[0]
    
    # Data type
    dtype = 'float16' if not torch.cuda.is_bf16_supported() else 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.cuda.amp.autocast(dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # Tokenizer
    if in_main_process: print("Preparing Tokenizer...")
    if "IndexHints" in cfg.task.train.dataset_cls:
        cfg.task.vocab = " ".join(list(map(str, range(int(cfg.task.max_position)+10))) + \
                                  [cfg.task.symbol, '='])
    tokenizer = build_tokenizer(cfg)
    if "IndexHints" in cfg.task.train.dataset_cls:
        id_index_hint_begin = tokenizer.token_to_id('10')
        id_index_hint_end = tokenizer.token_to_id(str(int(cfg.task.max_position)+9))

    # Random seed for data
    set_seed(seed=cfg.seed_data, device_type=device_type)

    # Dataset / Dataloader
    if in_main_process: print("Preparing Datasets...")
    dataset = build_dataset(cfg, verbose=in_main_process)
    if in_main_process: print("Preparing Dataloaders...")
    num_workers = cfg.training.num_workers
    sampler = {}
    for phase in dataset:
        sampler[phase] = DistributedSampler(
            dataset[phase], 
            shuffle=(phase == 'train'),
        )
    loader = build_loader(cfg, dataset, tokenizer, 
                          device if device_type=='cuda' else 'cpu', 
                          sampler, num_workers=num_workers)

    # Random seed for model & training
    set_seed(seed=cfg.seed, device_type=device_type)

    # Model
    model = build_model_from_scratch(cfg, tokenizer, device)
    if in_main_process:
        if getattr(cfg.model, 'd_positions', 1) == 1:
            model_summary = torchinfo.summary(model, (100,), batch_dim=0, dtypes=[torch.long], depth=5)
        else:
            model_summary = torchinfo.summary(model, depth=5)
        dict_cfg['total_params'] = model_summary.total_params
        dict_cfg['trainable_params'] = model_summary.trainable_params
    if device_type == 'cuda':
        if in_main_process: print("Preparing Distributed Model...")
        model = DDP(model, device_ids=[device_id], gradient_as_bucket_view=True, static_graph=True)    

    # Optimizer
    if in_main_process: print("Preparing Optimizer...")
    optimizer_kwargs =  DotMap(OmegaConf.to_container(cfg.training.optimizer))
    optimizer_type = optimizer_kwargs.pop('type')
    optimizer = ZeroRedundancyOptimizer(model.parameters(), getattr(torch.optim, optimizer_type), **optimizer_kwargs)
    # optimizer = getattr(torch.optim, optimizer_type)(model.parameters(), **optimizer_kwargs)

    # LR scheduler
    if in_main_process: print("Preparing Scheduler...")
    n_steps = cfg.training.n_steps
    n_epochs = math.ceil(n_steps/len(loader['train']))
    scheduler_kwargs = cfg.training.scheduler
    warmup_ratio = scheduler_kwargs.get('warmup_ratio', 0.1)
    try:
        scheduler = get_scheduler(
            scheduler_kwargs.type,
            optimizer,
            num_warmup_steps=int(warmup_ratio*n_steps),
            num_training_steps=n_steps
        )
    except ValueError:
        if scheduler_kwargs.type == 'custom_cosine':
            min_lr_ratio = scheduler_kwargs.get('min_lr_ratio', 0.1)
            scheduler = get_custom_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(warmup_ratio*n_steps),
                num_training_steps=n_steps,
                min_lr_ratio=min_lr_ratio
            )
        elif scheduler_kwargs.type == 'custom_linear':
            scheduler = get_custom_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(warmup_ratio*n_steps),
                num_training_steps=n_steps,
                min_lr_ratio=min_lr_ratio
            )
        else:
            raise ValueError(f"Undefined scheduler_kwargs.type='{scheduler_kwargs.type}'")

    # Training Misc
    model_name = cfg.model.model_name
    calc_acc_every_epochs = cfg.training.calc_acc_every_epochs
    log_every_steps = cfg.training.log_every_steps
    min_val_loss = 1e10
    grad_clip = cfg.training.grad_clip
    save = cfg.model.get('save', False)
    
    phases = list(loader.keys())  # e.g., ['train', 'val', 'val_hard', 'val_long', 'val_long_hard']
    if in_main_process:
        losses = {phase: [] for phase in phases}
        tokenwise_accuracies = {phase: [] for phase in phases}
        instancewise_accuracies = {phase: [] for phase in phases}
    
    ## Train! ##
    if in_main_process: print("Start Training")
    counter_training = 0
    for epoch in range(1, n_epochs+1):
        if counter_training >= n_steps: break
        loader['train'].sampler.set_epoch(epoch)
        for phase in phases:
            if phase != 'train' and not (epoch%calc_acc_every_epochs == 0 or epoch == n_epochs): continue
            if in_main_process: print(f"\nEpoch {epoch} {phase.upper()} begin at {now()}")
            start_time = time.time()
            pbar = loader[phase]
            loss_sum = 0.
            if epoch % calc_acc_every_epochs == 0 or epoch == 1:
                tokenwise_correct_sum = 0
                num_tokens_sum = 0
                instancewise_correct_sum = 0
            for batch_idx, model_inputs in enumerate(pbar, start=1):
                if "IndexHints" in cfg.task.train.dataset_cls and cfg.task.get('hide_index_hints', False):
                    model_inputs['labels'] = torch.where(
                        torch.logical_and(model_inputs['labels'] >= id_index_hint_begin, 
                                          model_inputs['labels'] <= id_index_hint_end),
                        -100,
                        model_inputs['labels']
                    )
                with torch.set_grad_enabled(phase == 'train'):
                    with ctx:
                        model_output = model(**model_inputs)
                        loss = model_output.loss
                    # Gradient update
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        if grad_clip > 0.:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        counter_training += 1
                # Logging in the middle of an epoch
                with torch.no_grad():
                    batchsize = len(model_inputs['input_ids'])
                    loss_sum += loss.float() * batchsize
                    if (batch_idx==1 or 
                        batch_idx % log_every_steps == 0 or 
                        batch_idx==len(loader[phase])):
                        print_training_update(phase, device, epoch, batch_idx, scheduler.get_last_lr()[0], loss.item(), start_time)
                    if epoch % calc_acc_every_epochs == 0 or epoch == n_epochs:
                        logits = model_output.logits
                        pred = torch.argmax(logits, dim=-1)
                        tokenwise_correct, num_tokens = get_tokenwise_accuracy(cfg, pred, model_inputs['labels'], tokenizer.pad_token_id, division=False)
                        instancewise_correct, _ = get_instancewise_accuracy(cfg, pred, model_inputs['labels'], tokenizer.pad_token_id, division=False)
                        tokenwise_correct_sum += tokenwise_correct
                        num_tokens_sum += num_tokens
                        instancewise_correct_sum += instancewise_correct

            # Logging at the end of an epoch
            all_reduce(loss_sum, op=ReduceOp.SUM)
            loss_avg = loss_sum.item()/len(dataset[phase])
            if in_main_process: losses[phase].append(loss_avg)
            if epoch % calc_acc_every_epochs == 0:
                all_reduce(tokenwise_correct_sum, op=ReduceOp.SUM)
                all_reduce(num_tokens_sum, op=ReduceOp.SUM)
                all_reduce(instancewise_correct_sum, op=ReduceOp.SUM)
                tokenwise_accuracy_avg = (tokenwise_correct_sum / num_tokens_sum).item()
                instancewise_accuracy_avg = instancewise_correct_sum.item() / len(dataset[phase])
                if in_main_process:
                    tokenwise_accuracies[phase].append(tokenwise_accuracy_avg)
                    instancewise_accuracies[phase].append(instancewise_accuracy_avg)
            if in_main_process:
                # Print result of epoch
                if epoch % calc_acc_every_epochs == 0:
                    print(f"Epoch {epoch}/{n_epochs} {phase.upper()} Loss {loss_avg:.6f} "
                        f"TokenAcc {tokenwise_accuracy_avg:.6f} "
                        f"InstAcc {instancewise_accuracy_avg:.6f}")
                else:
                    print(f"Epoch {epoch}/{n_epochs} {phase.upper()} Loss {loss_avg:.6f} ")
                # W&B
                if use_wandb:
                    log_data = {'loss': losses[phase][-1]}
                    if epoch % calc_acc_every_epochs == 0:
                        log_data['tokenwise_accuracy'] = tokenwise_accuracies[phase][-1]
                        log_data['instancewise_accuracy'] = instancewise_accuracies[phase][-1]
                    log_data = {f"{phase}/{k}": v for k, v in log_data.items()}
                    log_data['misc/learning_rate'] = scheduler.get_last_lr()[0]
                    wandb_run.log(log_data, step=counter_training)
                # plot results
                if phase == phases[-1] and epoch % calc_acc_every_epochs == 0:
                    fig, ax = plt.subplots(1,1)
                    for p in phases:
                        ax.plot(torch.arange(1,len(losses[p])+1).numpy()*calc_acc_every_epochs, 
                                losses[p], 
                                label=p+f" (Final:{losses[p][-1]:.3g} | Max:{max(losses[p]):.3g})", 
                                marker='.')
                    ax.legend()
                    ax.set_title("Loss")
                    fig.savefig(os.path.join(logging_path, f"loss.pdf"))
                    plt.close(fig)
                    fig, ax = plt.subplots(1,1)
                    for p in phases:
                        ax.plot(torch.arange(1,len(tokenwise_accuracies[p])+1).numpy()*calc_acc_every_epochs, 
                                tokenwise_accuracies[p], 
                                label=p+f" (Final:{tokenwise_accuracies[p][-1]:.3g} | Max:{max(tokenwise_accuracies[p]):.3g})", 
                                marker='.')
                    ax.legend()
                    ax.set_title("Tokenwise Accuracy")
                    fig.savefig(os.path.join(logging_path, f"tokenwise_accuracy.pdf"))
                    plt.close(fig)
                    fig, ax = plt.subplots(1,1)
                    for p in phases:
                        ax.plot(torch.arange(1,len(instancewise_accuracies[p])+1).numpy()*calc_acc_every_epochs,
                                instancewise_accuracies[p],
                                label=p+f" (Final:{instancewise_accuracies[p][-1]:.3g} | Max:{max(instancewise_accuracies[p]):.3g})",
                                marker='.')
                    ax.legend()
                    ax.set_title("Instance-wise Accuracy")
                    fig.savefig(os.path.join(logging_path, f"instancewise_accuracy.pdf"))
                    plt.close(fig)
                # Save Best Model (in terms of min val_long loss)
                if (cfg.model.get('save', False) and 
                    phase == 'val_long' and 
                    epoch % calc_acc_every_epochs == 0 and 
                    min_val_loss > losses['val_long'][-1]):
                    min_val_loss = losses['val_long'][-1]
                    torch.save(model.module.state_dict(), os.path.join(logging_path, f"best_{model_name}.pt"))

        # ## TRAIN ##
        # phase = 'train'
        # if in_main_process: print(f"\nEpoch {epoch} {phase.upper()} begin at {now()}")
        # start_time = time.time()
        # # Training Epoch
        # pbar = loader[phase]
        # loss_sum = 0.
        # tokenwise_correct_sum = 0
        # num_tokens_sum = 0
        # instancewise_correct_sum = 0
        # for batch_idx, model_inputs in enumerate(pbar, start=1):
        #     with ctx:
        #         model_output = model(**model_inputs)
        #         loss = model_output.loss
        #     scaler.scale(loss).backward()
        #     if grad_clip > 0.:
        #         scaler.unscale_(optimizer)
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        #     scaler.step(optimizer)
        #     scaler.update()
        #     optimizer.zero_grad(set_to_none=True)
        #     scheduler.step()
        #     counter_training += 1
        #     with torch.no_grad():
        #         # Logging in the middle of an epoch
        #         batchsize = len(model_inputs['input_ids'])
        #         loss_sum += loss.float() * batchsize
        #         if (batch_idx==1 or 
        #             batch_idx % log_every_steps == 0 or 
        #             batch_idx==len(loader[phase])):
        #             print_training_update(phase, device, epoch, batch_idx, scheduler.get_last_lr()[0], loss.item(), start_time)
        #         if epoch % calc_acc_every_epochs == 0 or epoch == n_epochs:
        #             logits = model_output.logits
        #             pred = torch.argmax(logits, dim=-1)
        #             tokenwise_correct, num_tokens = get_tokenwise_accuracy(cfg, pred, model_inputs['labels'], tokenizer.pad_token_id, division=False)
        #             instancewise_correct, _ = get_instancewise_accuracy(cfg, pred, model_inputs['labels'], tokenizer.pad_token_id, False)
        #             tokenwise_correct_sum += tokenwise_correct
        #             num_tokens_sum += num_tokens
        #             instancewise_correct_sum += instancewise_correct
        # # Logging at the end of an epoch
        # all_reduce(loss_sum, op=ReduceOp.SUM)
        # loss_avg = loss_sum.item()/len(dataset[phase])
        # if in_main_process: losses[phase].append(loss_avg)
        # if epoch % calc_acc_every_epochs == 0:
        #     all_reduce(tokenwise_correct_sum, op=ReduceOp.SUM)
        #     all_reduce(num_tokens_sum, op=ReduceOp.SUM)
        #     all_reduce(instancewise_correct_sum, op=ReduceOp.SUM)
        #     tokenwise_accuracy_avg = (tokenwise_correct_sum / num_tokens_sum).item()
        #     instancewise_accuracy_avg = instancewise_correct_sum.item() / len(dataset[phase])
        #     if in_main_process:
        #         tokenwise_accuracies[phase].append(tokenwise_accuracy_avg)
        #         instancewise_accuracies[phase].append(instancewise_accuracy_avg)
        # if in_main_process:
        #     if epoch % calc_acc_every_epochs == 0:
        #         print(f"Epoch {epoch}/{n_epochs} {phase.upper()} Loss {loss_avg:.6f} "
        #             f"TokenAcc {tokenwise_accuracy_avg:.6f} "
        #             f"InstAcc {instancewise_accuracy_avg:.6f}")
        #     else:
        #         print(f"Epoch {epoch}/{n_epochs} {phase.upper()} Loss {loss_avg:.6f} ")
        # if in_main_process:
        #     if use_wandb:
        #         log_data = {'loss': losses[phase][-1]}
        #         if epoch % calc_acc_every_epochs == 0:
        #             log_data['tokenwise_accuracy'] = tokenwise_accuracies[phase][-1],
        #             log_data['instancewise_accuracy'] = instancewise_accuracies[phase][-1],
        #         log_data = {f"{phase}/{k}": v for k, v in log_data.items()}
        #         log_data['misc/learning_rate'] = scheduler.get_last_lr()[0]
        #         wandb_run.log(log_data, step=counter_training)
            
        # ## VAL ##
        # phase = 'val'
        # if not (epoch%calc_acc_every_epochs == 0 or epoch == n_epochs): continue
        # if in_main_process: print(f"\nEpoch {epoch} {phase.upper()} begin at {now()}")
        # start_time = time.time()
        # # Training Epoch
        # pbar = loader[phase]
        # loss_sum = 0.
        # tokenwise_correct_sum = 0
        # num_tokens_sum = 0
        # instancewise_correct_sum = 0
        # for batch_idx, model_inputs in enumerate(pbar, start=1):
        #     with torch.no_grad():
        #         with ctx:
        #             model_output = model(**model_inputs)
        #             loss = model_output.loss
        #         # Logging in the middle of an epoch
        #         batchsize = len(model_inputs['input_ids'])
        #         loss_sum += loss.float() * batchsize
        #         if (batch_idx==1 or 
        #             batch_idx % log_every_steps == 0 or 
        #             batch_idx==len(loader[phase])):
        #             print_training_update(phase, device, epoch, batch_idx, scheduler.get_last_lr()[0], None, start_time)
        #         if epoch % calc_acc_every_epochs == 0 or epoch == n_epochs:
        #             logits = model_output.logits
        #             pred = torch.argmax(logits, dim=-1)
        #             tokenwise_correct, num_tokens = get_tokenwise_accuracy(cfg, pred, model_inputs['labels'], tokenizer.pad_token_id, division=False)
        #             instancewise_correct, _ = get_instancewise_accuracy(cfg, pred, model_inputs['labels'], tokenizer.pad_token_id, False)
        #             tokenwise_correct_sum += tokenwise_correct
        #             num_tokens_sum += num_tokens
        #             instancewise_correct_sum += instancewise_correct
        # # Logging at the end of an epoch
        # all_reduce(loss_sum, op=ReduceOp.SUM)
        # loss_avg = loss_sum.item()/len(dataset[phase])
        # if in_main_process: losses[phase].append(loss_avg)
        # if epoch % calc_acc_every_epochs == 0:
        #     all_reduce(tokenwise_correct_sum, op=ReduceOp.SUM)
        #     all_reduce(num_tokens_sum, op=ReduceOp.SUM)
        #     all_reduce(instancewise_correct_sum, op=ReduceOp.SUM)
        #     tokenwise_accuracy_avg = (tokenwise_correct_sum / num_tokens_sum).item()
        #     instancewise_accuracy_avg = instancewise_correct_sum.item() / len(dataset[phase])
        #     if in_main_process:
        #         tokenwise_accuracies[phase].append(tokenwise_accuracy_avg)
        #         instancewise_accuracies[phase].append(instancewise_accuracy_avg)
        # if in_main_process:
        #     if epoch % calc_acc_every_epochs == 0:
        #         print(f"Epoch {epoch}/{n_epochs} {phase.upper()} Loss {loss_avg:.6f} "
        #             f"TokenAcc {tokenwise_accuracy_avg:.6f} "
        #             f"InstAcc {instancewise_accuracy_avg:.6f}")
        #     else:
        #         print(f"Epoch {epoch}/{n_epochs} {phase.upper()} Loss {loss_avg:.6f} ")
        # if in_main_process:
        #     if use_wandb:
        #         log_data = {'loss': losses[phase][-1]}
        #         if epoch % calc_acc_every_epochs == 0:
        #             log_data['tokenwise_accuracy'] = tokenwise_accuracies[phase][-1],
        #             log_data['instancewise_accuracy'] = instancewise_accuracies[phase][-1],
        #         log_data = {f"{phase}/{k}": v for k, v in log_data.items()}
        #         wandb_run.log(log_data, step=counter_training)
            
        # ## VAL_LONG 
        # phase = 'val_long'
        # if in_main_process: print(f"\nEpoch {epoch} {phase.upper()} begin at {now()}")
        # start_time = time.time()
        # # Training Epoch
        # pbar = loader[phase]
        # loss_sum = 0.
        # tokenwise_correct_sum = 0
        # num_tokens_sum = 0
        # instancewise_correct_sum = 0
        # for batch_idx, model_inputs in enumerate(pbar, start=1):
        #     with torch.no_grad():
        #         with ctx:
        #             model_output = model(**model_inputs)
        #             loss = model_output.loss
        #         # Logging in the middle of an epoch
        #         batchsize = len(model_inputs['input_ids'])
        #         loss_sum += loss.float() * batchsize
        #         if (batch_idx==1 or 
        #             batch_idx % log_every_steps == 0 or 
        #             batch_idx==len(loader[phase])):
        #             print_training_update(phase, device, epoch, batch_idx, scheduler.get_last_lr()[0], None, start_time)
        #         if epoch % calc_acc_every_epochs == 0 or epoch == n_epochs:
        #             logits = model_output.logits
        #             pred = torch.argmax(logits, dim=-1)
        #             tokenwise_correct, num_tokens = get_tokenwise_accuracy(cfg, pred, model_inputs['labels'], tokenizer.pad_token_id, division=False)
        #             instancewise_correct, _ = get_instancewise_accuracy(cfg, pred, model_inputs['labels'], tokenizer.pad_token_id, False)
        #             tokenwise_correct_sum += tokenwise_correct
        #             num_tokens_sum += num_tokens
        #             instancewise_correct_sum += instancewise_correct
        # # Logging at the end of an epoch
        # all_reduce(loss_sum, op=ReduceOp.SUM)
        # loss_avg = loss_sum.item()/len(dataset[phase])
        # if in_main_process: losses[phase].append(loss_avg)
        # if epoch % calc_acc_every_epochs == 0:
        #     all_reduce(tokenwise_correct_sum, op=ReduceOp.SUM)
        #     all_reduce(num_tokens_sum, op=ReduceOp.SUM)
        #     all_reduce(instancewise_correct_sum, op=ReduceOp.SUM)
        #     tokenwise_accuracy_avg = (tokenwise_correct_sum / num_tokens_sum).item()
        #     instancewise_accuracy_avg = instancewise_correct_sum.item() / len(dataset[phase])
        #     if in_main_process:
        #         tokenwise_accuracies[phase].append(tokenwise_accuracy_avg)
        #         instancewise_accuracies[phase].append(instancewise_accuracy_avg)
        # if in_main_process:
        #     if epoch % calc_acc_every_epochs == 0:
        #         print(f"Epoch {epoch}/{n_epochs} {phase.upper()} Loss {loss_avg:.6f} "
        #             f"TokenAcc {tokenwise_accuracy_avg:.6f} "
        #             f"InstAcc {instancewise_accuracy_avg:.6f}")
        #     else:
        #         print(f"Epoch {epoch}/{n_epochs} {phase.upper()} Loss {loss_avg:.6f} ")
        # if in_main_process:
        #     if use_wandb:
        #         log_data = {'loss': losses[phase][-1]}
        #         if epoch % calc_acc_every_epochs == 0:
        #             log_data['tokenwise_accuracy'] = tokenwise_accuracies[phase][-1],
        #             log_data['instancewise_accuracy'] = instancewise_accuracies[phase][-1],
        #         log_data = {f"{phase}/{k}": v for k, v in log_data.items()}                    
        #         wandb_run.log(log_data, step=counter_training)
        #     if epoch % calc_acc_every_epochs == 0:
        #         fig, ax = plt.subplots(1,1)
        #         for p in phases:
        #             ax.plot(losses[p], label=p+f" (Final:{losses[p][-1]:.3g})", marker='.')
        #         ax.legend()
        #         ax.set_title("Loss")
        #         fig.savefig(os.path.join(logging_path, f"{model_name}_loss.pdf"))
        #         plt.close(fig)
        #         fig, ax = plt.subplots(1,1)
        #         for p in phases:
        #             ax.plot(torch.arange(len(tokenwise_accuracies[p])).numpy()*calc_acc_every_epochs, tokenwise_accuracies[p], label=p+f" (Final:{tokenwise_accuracies[p][-1]:.3g})", marker='.')
        #         ax.legend()
        #         ax.set_title("Tokenwise Accuracy")
        #         fig.savefig(os.path.join(logging_path, f"{model_name}_tokenwise_accuracy.pdf"))
        #         plt.close(fig)
        #         fig, ax = plt.subplots(1,1)
        #         for p in phases:
        #             ax.plot(torch.arange(len(instancewise_accuracies[p])).numpy()*calc_acc_every_epochs, instancewise_accuracies[p], label=p+f" (Final:{instancewise_accuracies[p][-1]:.3g})", marker='.')
        #         ax.legend()
        #         ax.set_title("Instance-wise Accuracy")
        #         fig.savefig(os.path.join(logging_path, f"{model_name}_instancewise_accuracy.pdf"))
        #         plt.close(fig)

        #     # Best Model Save (in terms of min val loss)
        #     if epoch % calc_acc_every_epochs == 0 and cfg.model.get('save', False) and min_val_loss > losses['val_long'][-1]:
        #         min_val_loss = losses['val_long'][-1]
        #         torch.save(model.module.state_dict(), os.path.join(logging_path, f"best_{model_name}.pt"))
    
    # After training
    if in_main_process:
        # W&B
        if use_wandb:
            wandb.finish()
        # Re-save configs
        dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        dict_cfg['loss'] = losses
        dict_cfg['tokenwise_accuracy'] = tokenwise_accuracies
        dict_cfg['instancewise_accuracy'] = instancewise_accuracies
        with open(os.path.join(logging_path, 'cfg.json'), 'w') as f:
            json.dump(dict_cfg, f, indent=2)
        # Save last model
        if cfg.model.get('save', False):
            torch.save(model.module.state_dict(), os.path.join(logging_path, f"last_{model_name}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--config_path', type=str,  default='./configs')
    parser.add_argument('--config_name', type=str,  default='config') 
    parser.add_argument('--overrides',   type=str,  default=[],     nargs='*')
    parser.add_argument('--device_ids',  type=int,  default=[0],    nargs='*') 
    parser.add_argument('--device_type', type=str,  default='cuda') 
    args = parser.parse_args()

    config_path = args.config_path
    config_name = args.config_name
    use_wandb = args.use_wandb
    overrides = args.overrides
    device_ids = args.device_ids
    device_type = args.device_type

    os.environ['PJRT_DEVICE'] = 'GPU'

    # Hydra Compose
    initialize(version_base=None, config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)

    logging_path = f"log/{cfg.group_name}/{cfg.exp_name}/seed{cfg.seed}_seedData{cfg.seed_data}"
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    # WandB
    dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if use_wandb:
        wandb_run = wandb.init(
            project=cfg.project_name, 
            entity=cfg.entity,
            config=dict_cfg,
            group=cfg.exp_name,
            reinit=True,
            settings=wandb.Settings(start_method="thread")
        ) 
    else:
        wandb_run = None
        with open(os.path.join(logging_path, 'cfg.json'), 'w') as f:
            json.dump(dict_cfg, f, indent=2)
    
    mp.spawn(mp_fn, args=(cfg, device_ids, device_type, logging_path, use_wandb, wandb_run), nprocs=len(device_ids))