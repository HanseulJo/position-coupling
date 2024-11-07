"""
Probing Attention (score) matrices of Customt5DecoderOnly model
"""

import argparse
from contextlib import nullcontext
from dotmap import DotMap
from hydra import compose, initialize
import json
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import torch
from tqdm import tqdm
from transformers import set_seed
from warnings import filterwarnings
filterwarnings("ignore")

from src.tokenization import build_tokenizer
from src.data import build_dataset, build_loader
from src.model import build_model_from_scratch, DECODER_BASED
from src.evaluate import get_tokenwise_accuracy, get_instancewise_accuracy

def evaluate(args):    
    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides
    n_digits = args.n_digits
    compile = args.compile
    title = args.title.replace("\\", "\n")

    # Hydra Compose
    initialize(version_base=None, config_path=config_path) 
    cfg = compose(config_name=config_name, overrides=overrides)

    # Bring model configs
    logging_path = os.path.join("log", cfg.group_name, cfg.exp_name, f"seed{cfg.seed}_seedData{cfg.seed_data}")
    with open(os.path.join(logging_path, "cfg.json")) as f:
        dict_cfg = json.load(f)
    for k in dict_cfg['model']:
        cfg.model[k] = dict_cfg['model'][k]
    
    # device
    if cfg.device=='cpu':
        device = torch.device('cpu')
    elif str(cfg.device).startswith('cuda:'):
        device = torch.device(cfg.device)

    # Data type
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if cfg.device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    
    # Training Misc
    model_name = cfg.model.model_name
    logging_path = f"log/{cfg.group_name}/{cfg.exp_name}/seed{cfg.seed}_seedData{cfg.seed_data}"
    print(logging_path)

    # Tokenizer
    tokenizer = build_tokenizer(cfg)
    vocab = dict(sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]))
    i2w = {v: k for k, v in vocab.items()}
    if cfg.task.bos_to_eos:
        i2w[vocab['[EOS]']] = 'BOS'
    
    # Model
    model = build_model_from_scratch(cfg, tokenizer, device)
    print("Number of params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if compile:
        model = torch.compile(model)  # compile!

    # get pretrained model
    model_path = os.path.join(logging_path, f'last_{model_name}.pt')
    mode = 'last'
    if cfg.get('best', False) or not os.path.exists(model_path):
        mode = 'best'
        print("Testing Best Model")
        model_path = os.path.join(logging_path, f'best_{model_name}.pt')
    if model_path.startswith(logging_path+'/last'):
        print("Testing Last Model")
    if not os.path.exists(model_path):
        print("no model:", model_path)
        return
    model.load_state_dict(torch.load(model_path, map_location=torch.device(cfg.device)))
    model.eval()

    folder = os.path.join(logging_path, 'heatmaps/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    if n_digits <= 2:
        wpe = model.decoder.wpe.weight.detach().cpu().numpy()[:cfg.task.max_position+1]
        wte = model.shared.weight.detach().cpu().numpy()
        annot = False
        for layer_idx, block in enumerate(model.decoder.block):
            Q = block.layer[0].SelfAttention.q.weight.detach().cpu().numpy()
            K = block.layer[0].SelfAttention.k.weight.detach().cpu().numpy()
            for head_idx, (q, k) in tqdm(enumerate(zip(np.split(Q, cfg.model.num_heads), np.split(K, cfg.model.num_heads))), total=cfg.model.num_heads):
                sns.set_theme(font_scale=0.1)
                sns.heatmap(q @ k.T, annot=annot, fmt='.1f')
                plt.savefig(f'{folder}/qk_layer{layer_idx}_head{head_idx}.pdf')
                plt.close()
                sns.set_theme(font_scale=0.006)
                sns.heatmap(wpe @ q.T @ k @ wpe.T / (cfg.model.d_kv**.5), annot=annot, fmt='.1f')
                plt.savefig(f'{folder}/wpeQKwpe_layer{layer_idx}_head{head_idx}.pdf')
                plt.close()
                sns.set_theme(font_scale=0.3)
                sns.heatmap(wte @ q.T @ k @ wte.T / (cfg.model.d_kv**.5), annot=annot, fmt='.1f', xticklabels=vocab.keys(), yticklabels=vocab.keys())
                plt.savefig(f'{folder}/wteQKwte_layer{layer_idx}_head{head_idx}.pdf')
                plt.close()
                sns.set_theme(font_scale=0.02)
                sns.heatmap(wte @ q.T @ k @ wpe.T / (cfg.model.d_kv**.5), annot=annot, fmt='.1f', yticklabels=vocab.keys())
                plt.savefig(f'{folder}/wteQKwpe_layer{layer_idx}_head{head_idx}.pdf')
                plt.close()
                sns.heatmap(wpe @ q.T @ k @ wte.T / (cfg.model.d_kv**.5), annot=annot, fmt='.1f', xticklabels=vocab.keys())
                plt.savefig(f'{folder}/wpeQKwte_layer{layer_idx}_head{head_idx}.pdf')
                plt.close()

    
    print(f"N={n_digits}")
    # sns.set_theme(font_scale=1.5/np.sqrt(n_digits))

    cfg.task.train.min_n_digits=1
    cfg.task.train.max_n_digits=1
    cfg.task.train.n_data=1
    cfg.task.val.min_n_digits=1
    cfg.task.val.max_n_digits=1
    cfg.task.val.n_data=1
    cfg.task.val_long.min_n_digits = n_digits
    cfg.task.val_long.max_n_digits = n_digits

    # Random seed
    set_seed(seed=999)

    # Dataset / Dataloader
    dataset = build_dataset(cfg)
    loader = build_loader(cfg, dataset, tokenizer, device)

    phase = 'val_long'

    # Epoch
    pbar = tqdm(loader[phase])
    num_items = 0
    instancewise_accuracy_sum = 0.
    att_score_sum = [0. for _ in range(cfg.model.num_layers)]
    att_probs_sum = [0. for _ in range(cfg.model.num_layers)]
    for batch_idx, model_inputs in enumerate(pbar):
        with ctx:
            model_output = model(output_attentions=True, **model_inputs)
        with torch.no_grad():
            logits = model_output.logits
            pred = torch.argmax(logits, dim=-1)
            if batch_idx == 0:
                idx = 0
                print("Input     :", model_inputs['input_ids'][idx])
                if 'position_ids' in model_inputs:
                    print("Position  :", model_inputs['position_ids'][idx])
                if cfg.model.model_name in DECODER_BASED:
                    lab = model_inputs['labels'][idx, 1:]
                    print("Prediction:", pred[idx, :-1][lab != -100])
                else:
                    lab = model_inputs['labels'][idx]
                    print("Prediction:", pred[idx][lab != -100])
                print("Label     :", lab[lab != -100])
            instancewise_accuracy, acc_arr = get_instancewise_accuracy(cfg, pred, model_inputs['labels'], tokenizer.pad_token_id, return_arr=True)
            acc = instancewise_accuracy.item() * 100
            batchsize = len(model_inputs['input_ids'])
            instancewise_accuracy_sum += instancewise_accuracy * batchsize
            # num_items += torch.sum(acc_arr).item()
            num_items += batchsize

        # if acc == 0:  continue

        for layer_idx in range(cfg.model.num_layers):
            # att_score_sum[layer_idx] += model_output.attentions[layer_idx]['scores'].float().detach()[acc_arr].cpu().numpy().sum(0)
            # att_probs_sum[layer_idx] += model_output.attentions[layer_idx]['probs'].float().detach()[acc_arr].cpu().numpy().sum(0)
            # att_score_sum[layer_idx] += model_output.attentions[layer_idx]['scores'].float().detach().cpu().numpy().sum(0)
            att_probs_sum[layer_idx] += model_output.attentions[layer_idx]['probs'].float().detach().cpu().numpy().sum(0)
                
    print("Averaged Attention matrix of", num_items, "items")
    if not os.path.exists(f'{folder}/{n_digits}digits_{mode}_{acc:.1f}/'):
        os.makedirs(f'{folder}/{n_digits}digits_{mode}_{acc:.1f}/')

    # cmap = sns.light_palette("orangered", as_cmap=True, reverse=True)
    cmap = sns.color_palette("YlOrBr_r", as_cmap=True)
    cmap.set_bad("black")
    for layer_idx in range(cfg.model.num_layers):
        ## Attention Score Matrices
        # att_heads = att_score_sum[layer_idx] / num_items
        # for head_idx, att in tqdm(enumerate(att_heads), total=cfg.model.num_heads, desc=f'layer{layer_idx} (score)...'):
        #     fig, ax = plt.subplots(1, 1, figsize=(8,8))
        #     sns.heatmap(att[:-1, :-1].T, annot=(n_digits<=10), fmt='.2f', vmin=att[att!=float('-inf')].min(), vmax=att.max(), cbar=False, ax=ax,
        #                 xticklabels=[i2w[w] for w in model_inputs['input_ids'][0].cpu().tolist()][:-1],
        #                 yticklabels=[i2w[w] for w in model_inputs['input_ids'][0].cpu().tolist()][:-1]
        #                 )
        #     ax.tick_params(labeltop=True, labelbottom = False, labelleft=False, labelright=True, labelsize=14, pad=0)
        #     fig.tight_layout()
        #     fig.savefig(f'{folder}/{n_digits}digits_{mode}_{acc:.1f}/scores_layer{layer_idx}_head{head_idx}.pdf')
        #     plt.close()

        ## Attention matrices (after softmax)
        att_heads = att_probs_sum[layer_idx] / num_items
        for head_idx, att in tqdm(enumerate(att_heads), total=cfg.model.num_heads, desc=f'layer{layer_idx} (probs)...'):
            # Full attention matrices
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            # ax.set_facecolor("black")
            sns.heatmap(att[:-1, :-1], annot=False, fmt='.2f', cbar=False, ax=ax,  vmin=0, vmax=1, mask=att[:-1, :-1]<0.01, cmap=cmap,
                        xticklabels=[i2w[w] for w in model_inputs['input_ids'][0].cpu().tolist()][:-1],
                        yticklabels=[i2w[w] for w in model_inputs['input_ids'][0].cpu().tolist()][:-1]
                        )
            ax.tick_params(labeltop=False, labelbottom = True, labelleft=True, labelright=False, labelsize=5, pad=0)
            plt.yticks(rotation=270)
            plt.xticks(rotation=0)
            # ax.get_xticklabels()[0].set_fontsize(10)
            # ax.get_yticklabels()[0].set_fontsize(10)
            ax.set_aspect('equal')
            ax.set_title(title.replace('?', str(layer_idx+1)).replace('!', str(head_idx+1)), weight="bold", fontsize=16)
            fig.tight_layout()
            fig.savefig(f'{folder}/{n_digits}digits_{mode}_{acc:.1f}/probs_layer{layer_idx}_head{head_idx}.pdf')
            plt.close()

            # Closed-up view
            # fig, ax = plt.subplots(1, 1, figsize=(9,3))
            # # ax.set_facecolor("black")
            # sns.heatmap(att[-n_digits-3:-1,:-1], annot=False, fmt='.2f', cbar=False, ax=ax, vmin=0, vmax=1, mask=att[-n_digits-3:-1,:-1]<0.01, cmap=cmap,
            #             xticklabels=[i2w[w] for w in model_inputs['input_ids'][0].cpu().tolist()][:-1],
            #             yticklabels=[i2w[w] for w in model_inputs['input_ids'][0].cpu().tolist()][-n_digits-3:-1]
            #             )
            # ax.tick_params(labeltop=True, labelbottom = False, labelleft=True, labelright=False, labelsize=14, pad=0)
            # plt.yticks(rotation=270)
            # ax.get_xticklabels()[0].set_fontsize(10)
            # fig.tight_layout()
            # fig.savefig(f'{folder}/{n_digits}digits_{mode}_{acc:.1f}/probs_layer{layer_idx}_head{head_idx}_reduced.pdf')
            # plt.close()
        
    instancewise_accuracy_avg = instancewise_accuracy_sum/len(dataset[phase])
    print("EM Acc:", instancewise_accuracy_avg.item() * 100) 
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_path', type=str,  default='./configs')
    parser.add_argument('--config_name', type=str,  default='config')
    parser.add_argument('--n_digits',type=int,  default=10)
    parser.add_argument('--compile',   action='store_true')
    parser.add_argument('--title',   type=str, default="")
    parser.add_argument('--overrides',   type=str,  default=[],  nargs='*')
    args = parser.parse_args()

    evaluate(vars(args))