from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

import torch
import numpy as np

BINARY_OPS = ['+', '*', '-', '/', '//', '%']

class SpecialToken:
    pad = "[PAD]"
    unk = "[UNK]"
    bos = "[BOS]"
    eos = "[EOS]"
    

def build_tokenizer(cfg):
    PAD = SpecialToken.pad
    UNK = SpecialToken.unk
    BOS = SpecialToken.bos
    EOS = SpecialToken.eos

    task_cfg = cfg.task
    symbol = task_cfg.symbol
    eos = task_cfg.eos
    bos_to_eos = task_cfg.get('bos_to_eos', False)
    padding_max_length = task_cfg.get('padding_max_length', None)

    ## Vocabulary ##
    vocab = {PAD: 0, UNK: 1}
    if not eos: EOS = PAD
    if bos_to_eos: BOS = EOS
    if BOS == SpecialToken.bos: vocab[BOS] = len(vocab)
    if EOS == SpecialToken.eos: vocab[EOS] = len(vocab)
    vocab_str = task_cfg.get('vocab', None)
    if vocab_str is None:
        # default vocab
        vocab.update({str(k): k+len(vocab) for k in range(10)})
        if isinstance(symbol, list):
            for s in symbol:
                vocab[s] = len(vocab)
        else:
            vocab[symbol] = len(vocab)
        vocab['='] = len(vocab)
    else:
        # custom vocab
        vocab_pre_size = len(vocab)
        vocab.update({str(w): i+vocab_pre_size for i, w in enumerate(vocab_str.split(' '))})
    
    ## Build Tokenizer ##
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=SpecialToken.unk))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])
    tokenizer.add_special_tokens(list(set([PAD, UNK, BOS] + ([EOS] if eos else []))))
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS} $A {EOS}" if eos else f"{BOS} $A",
        special_tokens=list(set(
            [(BOS, tokenizer.token_to_id(BOS))] + \
                ([(EOS, tokenizer.token_to_id(EOS))] if eos else [])
        ))
    )
    tokenizer.pad_token, tokenizer.pad_token_id = PAD, tokenizer.token_to_id(PAD)
    tokenizer.unk_token, tokenizer.unk_token_id = UNK, tokenizer.token_to_id(UNK)
    tokenizer.bos_token, tokenizer.bos_token_id = BOS, tokenizer.token_to_id(BOS)
    tokenizer.eos_token, tokenizer.eos_token_id = EOS, tokenizer.token_to_id(EOS)
    tokenizer.enable_padding(pad_id=tokenizer.pad_token_id, 
                             pad_token=tokenizer.pad_token,
                             length=padding_max_length)
    return tokenizer


## Batch-Tokenization Function (collator) - encoder-decoder ##
def tokenize_for_encoder_decoder(tokenizer, inputs, labels=None, input_positions=None, label_positions=None, device='cpu', arr_type='torch'):
    out = {}
    out['input_ids'] = torch.LongTensor([enc.ids for enc in tokenizer.encode_batch(inputs)]).to(device)
    if labels is not None:
        out['labels'] = torch.LongTensor([enc.ids[1:] for enc in tokenizer.encode_batch(labels)]).to(device)  # delete [BOS]
        out['labels'][out['labels'] == tokenizer.pad_token_id] = -100
    out['attention_mask'] = torch.ones_like(out['input_ids']).to(device)
    out['attention_mask'][out['input_ids'] == tokenizer.pad_token_id] = 0
    # out['decoder_input_ids'] = shift_tokens_right(out['labels'], tokenizer.pad_token_id, tokenizer.bos_token_id)
    # out['decoder_attention_mask'] = torch.ones_like(out['decoder_input_ids']).to(device)
    # out['decoder_attention_mask'][out['decoder_input_ids'] == tokenizer.pad_token_id] = 0
    # if labels is not None:
    #     out['labels'][out['labels'] == tokenizer.pad_token_id] = -100
    return out


## Batch-Tokenization Function (collator) - decoder ##
def tokenize_for_decoder(
        tokenizer: Tokenizer, 
        inputs, 
        labels, 
        input_positions=None, 
        label_positions=None, 
        arr_type='torch',
        device='cpu', 
    ):
    pad_token_id = tokenizer.token_to_id(SpecialToken.pad)

    out = {}
    concat = [f"{inp} = {lab}" for inp, lab in zip(inputs, labels)]
    out['input_ids'] = np.array([enc.ids for enc in tokenizer.encode_batch(concat)])

    # labels: it is -100 except for the label part of the sequence.
    # "tokenizer.eos_token == SpecialToken.eos" is equivalent to "cfg.task.eos == True"
    concat_inputs = [
        f"{inp}" + ("" if tokenizer.eos_token==SpecialToken.eos else " =") 
        for inp in inputs
    ]
    input_ids = np.array([enc.ids for enc in tokenizer.encode_batch(concat_inputs)]) # enc.ids includes EOS tokens
    padded_input_ids = np.concatenate(
        [input_ids, 
         np.full((input_ids.shape[0], out['input_ids'].shape[1]-input_ids.shape[1]),
                 pad_token_id)], 
        1
    )
    out['labels'] = np.where(
        np.logical_or(out['input_ids'] == pad_token_id, padded_input_ids != pad_token_id),
        -100,
        out['input_ids']
    )

    # attention mask: 0 if out['input_ids'] == pad_token_id, otherwise 1
    out['attention_mask'] = np.where(out['input_ids'] != pad_token_id, 1, 0)

    # position_ids
    if input_positions is not None and label_positions is not None:
        batchsize, total_length = out['input_ids'].shape
        if np.array(input_positions[0]).ndim == 1:
            position_ids = np.zeros((batchsize, total_length), dtype=int)
            for b, (inp_pos, lab_pos) in enumerate(zip(input_positions, label_positions)):
                pos = inp_pos + lab_pos
                position_ids[b, 1:1+len(pos)] = np.array(pos, dtype=int)
        elif np.array(input_positions[0]).ndim == 2:  # multi dimensional position ids
            position_id_dim = len(input_positions[0])
            position_ids = np.zeros((position_id_dim, batchsize, total_length), dtype=int)
            for b, (inp_positions, lab_positions) in enumerate(zip(input_positions, label_positions)):
                for d, (inp_pos, lab_pos) in enumerate(zip(inp_positions, lab_positions)):
                    pos = inp_pos + lab_pos
                    position_ids[d, b, 1:1+len(pos)] = np.array(pos, dtype=int)
        out['position_ids'] = position_ids

    if arr_type == 'torch':
        # torch Tensor
        for k in out:
            out[k] = torch.LongTensor(out[k]).to(device)
    
    return out


if __name__ == "__main__":
    from dotmap import DotMap

    cfg = DotMap(task=DotMap(symbol='+',eos=False, bos_to_eos=False))
    tokenizer = build_tokenizer(cfg)
    print(tokenizer.token_to_id("="))

    inputs = ["1 2 + 3 4", "1 2 3 4 + 5 6 7 8"]
    labels = ["4 6", "6 9 1 2"]
    out = tokenize_for_decoder(tokenizer, inputs, labels)
    print(out['input_ids'])
    print(out['labels'])
    print(out['attention_mask'])
