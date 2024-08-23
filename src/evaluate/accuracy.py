import torch

from src.model import DECODER_BASED

## Evaluate accuracy ##
def get_tokenwise_accuracy(cfg, predictions, references, pad_token_id, division=True):
    device = predictions.device
    references = references.to(device)
    if cfg.model.model_name in DECODER_BASED:
        predictions = predictions[..., :-1]
        references = references[..., 1:]
    pad_mask = torch.logical_and(references != pad_token_id, references != -100)
    acc_mask = predictions == references
    mask = torch.logical_and(acc_mask, pad_mask)
    correct = mask.sum()
    samples = pad_mask.sum()
    if division:
        accuracy = correct / samples
        return accuracy
    else:
        return correct, samples

def get_instancewise_accuracy(cfg, predictions, references, pad_token_id, division=True, return_arr=False):
    device = predictions.device
    references = references.to(device)
    if cfg.model.model_name in DECODER_BASED:
        predictions = predictions[..., :-1]
        references = references[..., 1:]
    pad_mask = torch.logical_or(references == pad_token_id, references == -100)
    acc_mask = predictions == references
    mask = torch.logical_or(acc_mask, pad_mask)
    acc = torch.sum(mask, dim=1) == references.size(1)
    correct = torch.sum(acc)
    samples = len(acc)
    accuracy = correct / samples
    if division:
        if return_arr:
            return accuracy, acc
        return accuracy
    else:
        if return_arr:
            return correct, samples, acc
        return correct, samples
    
def get_parity_accuracy(cfg, predictions, references, pad_token_id, division=True, return_arr=False):
    device = predictions.device
    references = references.to(device)
    if cfg.model.model_name in DECODER_BASED:
        predictions = predictions[..., :-1]
        references = references[..., 1:]
    # pad_mask = torch.logical_or(references == pad_token_id, references == -100)
    # acc_mask = predictions == references
    # mask = torch.logical_or(acc_mask, pad_mask)
    acc = torch.stack([
        torch.all(pred[ref != -100][-2:-1] == ref[ref != -100][-2:-1])
        for pred, ref in zip(predictions, references)
    ]).to(device)
    correct = torch.sum(acc)
    samples = len(acc)
    accuracy = correct / samples
    if division:
        if return_arr:
            return accuracy, acc
        return accuracy
    else:
        if return_arr:
            return correct, samples, acc
        return correct, samples