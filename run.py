"""PyTorch-based Runner"""

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
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import torch
import torchinfo
from tqdm import tqdm
from transformers import get_scheduler
import wandb
from warnings import filterwarnings
filterwarnings("ignore")

from src.tokenization import build_tokenizer
from src.data import build_dataset, build_loader
from src.model import build_model_from_scratch, DECODER_BASED
from src.training import get_custom_cosine_schedule_with_warmup, get_custom_linear_schedule_with_warmup, set_seed
from src.evaluate import get_tokenwise_accuracy, get_instancewise_accuracy
from src.common import print_example, print_2D


def run(args):
    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    # Hydra Compose
    initialize(version_base=None, config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)

    logging_path = os.path.join("log", cfg.group_name, cfg.exp_name, f"seed{cfg.seed}_seedData{cfg.seed_data}")
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    # WandB
    dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    use_wandb = cfg.use_wandb
    if use_wandb:
        run = wandb.init(
            project=cfg.project_name, 
            entity=cfg.entity,
            config=dict_cfg,
            group=cfg.exp_name,
            reinit=True,
            settings=wandb.Settings(start_method="thread")
        ) 
    else:
        with open(os.path.join(logging_path, 'cfg.json'), 'w') as f:
            json.dump(dict_cfg, f, indent=2)

    # device
    if cfg.device=='cpu':
        device = torch.device('cpu')
        device_type = 'cpu'
    elif str(cfg.device).startswith('cuda:'):
        device = torch.device(cfg.device)
        device_type = 'gpu'
    
    # Data type & device
    dtype = 'float16' if not torch.cuda.is_bf16_supported() else 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    if device_type == 'gpu':
        ctx = torch.cuda.amp.autocast(dtype=ptdtype)
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    else:
        ctx = nullcontext()
        scaler = None
    
    # Tokenizer
    if "IndexHints" in cfg.task.train.dataset_cls:
        cfg.task.vocab = " ".join(list(map(str, range(int(cfg.task.max_position)+10))) + \
                                  [cfg.task.symbol, '='])
    tokenizer = build_tokenizer(cfg)
    if "IndexHints" in cfg.task.train.dataset_cls:
        id_index_hint_begin = tokenizer.token_to_id('10')
        id_index_hint_end = tokenizer.token_to_id(str(int(cfg.task.max_position)+9))

    # Random seed for data
    set_seed(seed=cfg.seed_data)

    # Dataset / Dataloader
    dataset = build_dataset(cfg)
    loader = build_loader(cfg, dataset, tokenizer, device)

    # Random seed for model & training
    set_seed(seed=cfg.seed)

    # Model
    model = build_model_from_scratch(cfg, tokenizer, device)
    if getattr(cfg.model, 'd_positions', 1) == 1:
        model_summary = torchinfo.summary(model, (100,), batch_dim=0, dtypes=[torch.long], depth=5)
    else:
        model_summary = torchinfo.summary(model, depth=5)
    dict_cfg['total_params'] = model_summary.total_params
    dict_cfg['trainable_params'] = model_summary.trainable_params

    # Optimizer
    optimizer_kwargs =  DotMap(OmegaConf.to_container(cfg.training.optimizer))
    optimizer_type = optimizer_kwargs.pop('type')
    optimizer = getattr(torch.optim, optimizer_type)(model.parameters(), **optimizer_kwargs)

    # LR scheduler
    n_steps = cfg.training.n_steps
    n_epochs = math.ceil(n_steps/len(loader['train']))
    scheduler_kwargs = cfg.training.scheduler
    warmup_ratio = scheduler_kwargs.warmup_ratio
    try:
        scheduler = get_scheduler(
            scheduler_kwargs.type,
            optimizer,
            num_warmup_steps=int(warmup_ratio*n_steps),
            num_training_steps=n_steps
        )
    except ValueError:
        min_lr_ratio = scheduler_kwargs.min_lr_ratio
        if scheduler_kwargs.type == 'custom_cosine':
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
    min_val_loss = 1e10
    grad_clip = cfg.training.grad_clip
    save = cfg.model.get('save', False)

    phases = list(loader.keys())  # ['train', 'val', 'val_long']
    if use_wandb:
        columns = ["Input", "Label", "DecodedOutput"]
        text_table = {phase: wandb.Table(columns=columns) for phase in phases}
    
    losses = {phase: [] for phase in phases}
    tokenwise_accuracies = {phase: [] for phase in phases}
    instancewise_accuracies = {phase: [] for phase in phases}
    
    ## Train! ##
    counter_training = 0
    for epoch in range(1, n_epochs+1):
        if counter_training >= n_steps: break
        for phase in phases:
            if phase != 'train' and not (epoch%calc_acc_every_epochs == 0 or epoch == n_epochs): continue
            # Training Epoch
            pbar = tqdm(loader[phase])
            loss_sum = 0.
            if epoch % calc_acc_every_epochs == 0 or epoch == 1:
                tokenwise_correct_sum = 0
                num_tokens_sum = 0
                instancewise_correct_sum = 0
            for batch_idx, model_inputs in enumerate(pbar):
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
                    if phase == 'train' and epoch > 0:
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            if grad_clip > 0.:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            if grad_clip > 0.:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        counter_training += 1
                with torch.no_grad():
                    batchsize = len(model_inputs['input_ids'])
                    loss_sum += loss.float() * batchsize
                    if not use_wandb and batch_idx == 0:
                        logits = model_output.logits
                        pred = torch.argmax(logits, dim=-1)
                        id_0 = tokenizer.token_to_id('0')
                        d_positions = getattr(cfg.model, 'd_positions', None)
                        print(phase.upper())
                        if d_positions is None:
                            print("Input     :", model_inputs['input_ids'][0].cpu().numpy() - id_0)
                            if 'position_ids' in model_inputs:
                                print("Position  :", model_inputs['position_ids'][0].cpu().numpy())
                            if model_name in DECODER_BASED:
                                lab = model_inputs['labels'][0, 1:]
                                print("Prediction:", pred[0, :-1][lab != -100].cpu().numpy() - id_0)
                            else:
                                lab = model_inputs['labels'][0]
                                print("Prediction:", pred[0][lab != -100].cpu().numpy() - id_0)
                            print("Label     :", lab[lab != -100].cpu().numpy() - id_0)
                        elif not cfg.task.train.dataset_cls.startswith("MineSweeper"):
                            print("Input     :", model_inputs['input_ids'][0].cpu().numpy() - id_0)
                            if 'position_ids' in model_inputs:
                                print("Position  :", model_inputs['position_ids'][:, 0].cpu().numpy())
                            if model_name in DECODER_BASED:
                                lab = model_inputs['labels'][0, 1:]
                                print("Prediction:", pred[0, :-1][lab != -100].cpu().numpy() - id_0)
                            else:
                                lab = model_inputs['labels'][0]
                                print("Prediction:", pred[0][lab != -100].cpu().numpy() - id_0)
                            print("Label     :", lab[lab != -100].cpu().numpy() - id_0)
                        else: # e.g. Minesweeper with coupling
                            assert model_name in DECODER_BASED
                            print_2D(model_inputs, pred, id_0)
                    if epoch % calc_acc_every_epochs == 0 or epoch == n_epochs:
                        logits = model_output.logits
                        pred = torch.argmax(logits, dim=-1)
                        tokenwise_correct, num_tokens = get_tokenwise_accuracy(cfg, pred, model_inputs['labels'], tokenizer.pad_token_id, division=False)
                        instancewise_correct, _ = get_instancewise_accuracy(cfg, pred, model_inputs['labels'], tokenizer.pad_token_id, division=False)
                        tokenwise_correct_sum += tokenwise_correct
                        num_tokens_sum += num_tokens
                        instancewise_correct_sum += instancewise_correct
                        pbar.set_description(f"[{counter_training}/{n_steps}] {phase.upper()} | LR:{scheduler.get_last_lr()[0]:.3g} | Loss:{loss:.3f} | "
                                            f"TokenAcc:{tokenwise_correct/num_tokens:.3f} | InstAcc:{instancewise_correct/batchsize:.3f}") 
                    else:
                        pbar.set_description(f"[{counter_training}/{n_steps}] {phase.upper()} | LR:{scheduler.get_last_lr()[0]:.3g} | Loss:{loss:.3f}")        
            # Logging at the end of epoch
            loss_avg = loss_sum.item()/len(dataset[phase])
            losses[phase].append(loss_avg)
            if epoch % calc_acc_every_epochs == 0:
                tokenwise_accuracy_avg = (tokenwise_correct_sum/num_tokens_sum).item()
                instancewise_accuracy_avg = instancewise_correct_sum.item()/len(dataset[phase])
                tokenwise_accuracies[phase].append(tokenwise_accuracy_avg)
                instancewise_accuracies[phase].append(instancewise_accuracy_avg)
                if getattr(cfg.model, 'd_positions', None) is None:
                    _, _, _, example = print_example(cfg, ctx, epoch, phase, tokenizer, dataset, model, verbose=False)
                    print(example)
            # W&B
            if use_wandb:
                log_data = {'loss': loss_avg}
                if epoch % calc_acc_every_epochs == 0:
                    log_data['tokenwise_accuracy'] = tokenwise_accuracy_avg
                    log_data['instancewise_accuracy'] = instancewise_accuracy_avg
                    if getattr(cfg.model, 'd_positions', None) is None: 
                        log_data['example'] = example
                log_data = {f"{phase}/{k}": v for k, v in log_data.items()}
                if phase == 'train':
                    log_data['misc/learning_rate'] = scheduler.get_last_lr()[0]
                run.log(log_data, step=counter_training)
            # Print result of epoch
            if epoch % calc_acc_every_epochs == 0:
                print(f"Epoch {epoch}/{n_epochs} {phase.upper()} Loss {loss_avg:.6f} "
                      f"TokenAcc {tokenwise_accuracy_avg:.6f} "
                      f"InstAcc {instancewise_accuracy_avg:.6f}")
            else:
                print(f"Epoch {epoch}/{n_epochs} {phase.upper()} Loss {loss_avg:.6f} ")
            # Plot results
            if phase == phases[-1] and epoch % calc_acc_every_epochs == 0:
                fig, ax = plt.subplots(1,1) 
                for p in phases:
                    ax.semilogy(torch.arange(1,len(losses[p])+1).numpy()*calc_acc_every_epochs, 
                            losses[p], 
                            label=p+f" (Final:{losses[p][-1]:.3g} | Min:{min(losses[p]):.3g})", 
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
            if save and phase == 'val_long' and min_val_loss > losses['val_long'][-1]:
                min_val_loss = losses['val_long'][-1]
                torch.save(model.state_dict(), os.path.join(logging_path, f"best_{model_name}.pt"))
        
        print()
    
    # Finish W&B
    if use_wandb:
        if getattr(cfg.model, 'd_positions', None) is None:
            for phase in phases:
                input_str, label_str, decoded, _ = print_example(cfg, ctx, epoch, phase, tokenizer, dataset, model, verbose=not use_wandb)
                text_table[phase].add_data(input_str, label_str, decoded)
                run.log({f"{phase}_text_table": text_table[phase]})

        wandb.finish()
    
    # Save config
    dict_cfg['best_val_long_loss'] = min_val_loss
    dict_cfg['loss'] = losses
    dict_cfg['tokenwise_accuracy'] = tokenwise_accuracies
    dict_cfg['instancewise_accuracy'] = instancewise_accuracies
    with open(os.path.join(logging_path, 'cfg.json'), 'w') as f:
        json.dump(dict_cfg, f, indent=2)
    
    # Save last model
    if save:
        torch.save(model.state_dict(), os.path.join(logging_path, f"last_{model_name}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_path', type=str,    default='./configs')
    parser.add_argument('--config_name', type=str,    default='config') 
    parser.add_argument('--overrides',   action='append', default=[])
    args = parser.parse_args()

    run(vars(args))