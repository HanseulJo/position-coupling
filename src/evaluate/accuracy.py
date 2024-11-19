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


def get_parity_accuracy(cfg, predictions, references, eos_token_id, division=True, return_arr=False):
    device = predictions.device
    references = references.to(device)
    if cfg.model.model_name in DECODER_BASED:
        predictions = predictions[..., :-1]
        references = references[..., 1:]
    
    samples = references.size(0)

    # compute is_answer: True if a token before eos, but not for query token
    is_eos_ref = references == eos_token_id
    is_response_ref = torch.cummax((references != -100).long(), dim=1).values == 1
    is_answer_ref = torch.cat([is_eos_ref[:, 1:], is_eos_ref[:, :1]], dim=1)
    is_answer_ref = torch.logical_and(is_answer_ref, is_response_ref)
    is_eos_pred = predictions == eos_token_id
    is_answer_pred = torch.cat([is_eos_pred[:, 1:], is_eos_pred[:, :1]], dim=1)
    is_answer_pred = torch.logical_and(is_answer_pred, is_response_ref)
    
    acc = []
    for pred, ref, pred_mask, ref_mask in zip(predictions, references, is_answer_pred, is_answer_ref):
        p = pred[pred_mask]
        r = ref[ref_mask]
        if p.size(0) == 0: p = pred[-1:]
        acc.append((p[0]==r[0]).item())
    
    correct = torch.tensor(acc).sum()
    accuracy = correct / samples
    if division:
        if return_arr:
            return accuracy, acc
        return accuracy
    else:
        if return_arr:
            return correct, samples, acc
        return correct, samples

def get_answerwise_accuracy(cfg, predictions, references, eos_token_id, sep_token_id, division=True, return_arr=False):
    device = predictions.device
    references = references.to(device)
    if cfg.model.model_name in DECODER_BASED:
        predictions = predictions[..., :-1]
        references = references[..., 1:]

    samples = references.size(0)

    # computing is_response: True for all token after "=". Including paddings!
    is_response_ref = torch.cummax((references != -100).long(), dim=1).values == 1

    # computing eos_mask: True for all tokens before the first eos
    # given [_query_, =, _noneos_, eos, _any_],
    # will have [False, ..., False, True, ..., True, False, ..., False], where True's indicate _noneos_.
    is_eos_ref = torch.logical_and(references == eos_token_id, is_response_ref)
    is_eos_ref_cummax, _ = torch.cummax(is_eos_ref.long(), dim=1)
    eos_mask_ref = torch.logical_and(is_eos_ref_cummax == 0, is_response_ref)
    is_eos_pred = torch.logical_and(predictions == eos_token_id, is_response_ref)
    is_eos_pred_cummax, _ = torch.cummax(is_eos_pred.long(), dim=1)
    eos_mask_pred = torch.logical_and(is_eos_pred_cummax == 0, is_response_ref)

    # computing sep_mask: True for all tokens from the last sep
    # given [_any_, sep, _nonsep_], 
    # will have [False, ..., False, True, ..., True], where True's indicate the last sep & _nonsep_.
    is_sep_ref = torch.logical_and(references == sep_token_id, eos_mask_ref)
    is_sep_ref_cumsum = torch.cumsum(is_sep_ref.long(), dim=1)
    sep_mask_ref = is_sep_ref_cumsum == is_sep_ref_cumsum[:, -1:]
    is_sep_pred = torch.logical_and(predictions == sep_token_id, eos_mask_pred)
    is_sep_pred_cumsum = torch.cumsum(is_sep_pred.long(), dim=1)
    sep_mask_pred = is_sep_pred_cumsum == is_sep_pred_cumsum[:, -1:]

    # computing is_answer: True for the answer tokens (no sep, no eos)
    is_answer_ref = torch.logical_and(sep_mask_ref, eos_mask_ref)
    is_answer_pred = torch.logical_and(sep_mask_pred, eos_mask_pred)

    acc = []
    for pred, ref, pred_mask, ref_mask in zip(predictions, references, is_answer_pred, is_answer_ref):
        p = pred[pred_mask]
        r = ref[ref_mask]
        acc.append(p.size() == r.size() and (p==r).all())
    
    correct = torch.tensor(acc).sum()
    accuracy = correct / samples
    if division:
        if return_arr:
            return accuracy, acc
        return accuracy
    else:
        if return_arr:
            return correct, samples, acc
        return correct, samples
