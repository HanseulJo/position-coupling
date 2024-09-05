import math
import torch
from torch import nn
from torch.nn import functional as F

POSITION_ENCODING_REL_T5_BIAS = "t5_relative_bias"
POSITION_ENCODING_REL_TRANSFORMER_XL = "transformer_xl_relative_encoding"
POSITION_ENCODING_ROTARY = "rotary"
POSITION_ENCODING_ROTARY_RERUN = "rotary_rerun"
POSITION_ENCODING_ROTARY_NEW = "new_rotary"
POSITION_ENCODING_ABS_LEARNED = "abs_learned"
POSITION_ENCODING_ABS_SINUSOID = "abs_sinusoid"
POSITION_ENCODING_ALiBi = "alibi"
POSITION_ENCODING_ALiBi_LEARNED = "alibi_learned"
POSITION_ENCODING_NONE = "none"
POSITION_ENCODING_FIRE = "fire"
POSITION_ENCODING_COUPLED_REL_BIAS = "coupled_relative_bias"

def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq)
        .to(x.device)
        .float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    """
    Example: [a, b, c, d] -> [-b, a, -d, c]
    """
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(
        lambda t: t[None, offset : x.shape[1] + offset, None, :].repeat_interleave(
            2, 3
        ),
        sincos,
    )
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


def apply_rotary_pos_emb_new(x, sincos, offset=0):
    sin, cos = map(
        lambda t: t[:, :, None, :].repeat_interleave(2, 3),
        sincos,
    )
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class FixedAbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(16384).type_as(inv_freq)
        sinusoid_inp = torch.einsum("i , j -> i j", t, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.embed = nn.Embedding.from_pretrained(emb, freeze=True)

    def forward(self, position_ids: torch.Tensor):
        return self.embed(position_ids.long())


class FixedRotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, rotary_dim: int, rotary_base: int = 10000, max_position: int = 16384
    ):
        super().__init__()
        # This is an inverse frequency tensor
        # Each dimension has a higher denominator than the previous one
        # So, the frequency will be lower for higher dimensions
        inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
        )  # [rotary_dim/2]

        # Now, we create frequencies for each position
        t = torch.arange(max_position, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_position, rotary_dim/2]

        sins = torch.sin(freqs)
        coss = torch.cos(freqs)

        emb = torch.cat([sins, coss], dim=-1)  # [max_position, rotary_dim]
        self.embed = nn.Embedding.from_pretrained(emb, freeze=True)

    def forward(self, position_ids: torch.Tensor):
        return self.embed(position_ids.long())
    

def build_alibi_tensor(
    attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.
    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        device=attention_mask.device,
        dtype=torch.float32,
    )
    powers = torch.arange(
        1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32
    )
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            device=attention_mask.device,
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1,
            1 + 2 * num_remaining_heads,
            2,
            device=attention_mask.device,
            dtype=torch.int32,
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


class Fire(nn.Module):
    def __init__(self, d_hidden, num_heads):
        super().__init__()
        self.d_hidden = d_hidden
        self.num_heads = num_heads

        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.d_hidden, bias=False),
                nn.ReLU(),
                nn.Linear(self.d_hidden, self.d_hidden, bias=False),
                nn.ReLU(),
                nn.Linear(self.d_hidden, 1, bias=False)
            ) for _ in range(self.num_heads)
        ])

    def forward(self, x):
        x = x.unsqueeze(-1)
        output = torch.cat([
            module(x) for module in self.nets
        ], dim=-1)
        return output
