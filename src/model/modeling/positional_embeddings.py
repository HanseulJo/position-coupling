import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
import logging

logger = logging.getLogger(__name__)

POSITION_ENCODING_REL_T5_BIAS = "t5_relative_bias"
POSITION_ENCODING_REL_TRANSFORMER_XL = "transformer_xl_relative_encoding"
POSITION_ENCODING_ROTARY_OLD = "rotary_old"
POSITION_ENCODING_ROTARY_NEW = "rotary_new"
POSITION_ENCODING_ROTARY_DEFAULT = "rotary_default"
POSITION_ENCODING_ROTARY_DEFAULT_HIROPE = "rotary_default_hirope"
POSITION_ENCODING_ROTARY_DEFAULT_MULTIROPE = "rotary_default_multirope"
POSITION_ENCODING_ROTARY_LINEAR = "rotary_linear"
POSITION_ENCODING_ROTARY_DYNAMIC = "rotary_dynamic"
POSITION_ENCODING_ROTARY_YARN = "rotary_yarn"
POSITION_ENCODING_ROTARY_LONGROPE = "rotary_longrope"
POSITION_ENCODING_ROTARY_LLAMA3 = "rotary_llama3"
POSITION_ENCODING_ABS_LEARNED = "abs_learned"
POSITION_ENCODING_ABS_SINUSOID = "abs_sinusoid"
POSITION_ENCODING_ALIBI = "alibi"
POSITION_ENCODING_ALIBI_LEARNED = "alibi_learned"
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


def apply_rotary_pos_emb_old(x, sincos, offset=0):
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
    


def rotate_half(x):
    """Brought from `transformers.models.llama`.
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Brought from `transformers.models.llama`.
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def get_inv_freq_for_hierarchical_rope(inv_freq, d_positions, config):
    division_ratios = list(getattr(config, 'division_ratios', [(i+1)/d_positions for i in range(d_positions)]))
    rotary_dim_half = inv_freq.shape[0]
    div = [0] + [round(ratio * rotary_dim_half) for ratio in division_ratios]
    inv_freq = torch.block_diag(*(inv_freq[a:b] for a, b in zip(div[:-1], div[1:])))
    return inv_freq

def get_inv_freq_for_multidimensional_rope(inv_freq, d_positions, config):
    division_ratios = list(getattr(config, 'division_ratios', [(i+1)/d_positions for i in range(d_positions)]))
    rotary_dim_half = inv_freq.shape[0]
    div = [0] + [round(ratio * rotary_dim_half) for ratio in division_ratios]
    partitioned = [inv_freq[a:b] for a, b in zip(div[:-1], div[1:])]
    inv_freq_list = []
    for _ in range(d_positions):
        inv_freq_list.append(torch.cat(partitioned))
        partitioned = partitioned[-1:] + partitioned[:-1]  # cyclic rotation of partitioned chunks
    inv_freq = torch.stack(inv_freq_list)
    return inv_freq

ROPE_MULTIPOS_FUNCTIONS = {
    'hirope': get_inv_freq_for_hierarchical_rope,
    'multirope': get_inv_freq_for_multidimensional_rope
}

class RotaryEmbedding(nn.Module):
    """Brought from `transformers.models.llama.LlamaRotaryEmbedding`."""
    def __init__(
        self,
        rope_type="default",
        d_positions=None,
        config=None,
    ):
        super().__init__()
        if len(rope_type.split('_')) == 2:
            self.rope_type, self.multipos_type = rope_type.split('_')
        else:
            self.rope_type, self.multipos_type = rope_type, None
        self.d_positions = d_positions
        self.max_seq_len_cached = config.n_positions
        self.original_max_seq_len = config.n_positions

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config)

        if self.d_positions is not None and self.multipos_type is not None:
            self.rope_multipos_fn = ROPE_MULTIPOS_FUNCTIONS[self.multipos_type]
            inv_freq = self.rope_multipos_fn(inv_freq, self.d_positions, config)

        self.register_buffer("inv_freq", inv_freq, persistent=False)  # (rope_dim,) where rope_dim = int(config.head_dim * config.partial_rotary_factor)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            if self.d_positions is not None and self.multipos_type is not None:
                self.rope_multipos_fn = ROPE_MULTIPOS_FUNCTIONS[self.multipos_type]
                inv_freq = self.rope_multipos_fn(inv_freq, self.d_positions, self.config)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, position_ids):
        device = position_ids.device
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=device)

        # Core RoPE block
        batchsize = position_ids.shape[-2]
        if self.d_positions is None:
            inv_freq_expanded = self.inv_freq[None, :, None].float().to(device).expand(batchsize, -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
        else:
            inv_freq_expanded = self.inv_freq[:, None, :, None].float().to(device).expand(-1, batchsize, -1, 1)
            position_ids_expanded = position_ids[:, :, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(-2, -1)  # outer product
            if self.d_positions is not None:
                assert freqs.ndim == 4
                freqs = freqs.sum(0)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos, sin
    

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


class FIRE(nn.Module):
    def __init__(self, num_heads, mlp_hidden, c0=0.1, L0_sqrt=16., eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.mlp_hidden = mlp_hidden
        self.eps = eps

        self.net = nn.Sequential(
            nn.Linear(1, self.mlp_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, self.mlp_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, self.num_heads, bias=False)
        )

        self.c = nn.Parameter(torch.tensor(c0))
        self.L_sqrt = nn.Parameter(torch.tensor(L0_sqrt))

        self.cached_matrix = None
        self.cached_seq_len = None

    def forward(self, x: torch.Tensor, q_pos=None, k_pos=None):
        # x : (batch_size, n_heads, query_length, key_length)
        # q_pos : (batch_size, query_length)
        # k_pos : (batch_size, key_length)
        seq_len_q = x.size(-2)
        seq_len_k = x.size(-1)
        
        if self.cached_seq_len != seq_len_k or (q_pos is not None and k_pos is not None):
            if q_pos is None or k_pos is None:
                q_pos = torch.arange(seq_len_q, device=x.device)[None,:]  # i
                k_pos = torch.arange(seq_len_k, device=x.device)[None,:]  # j
            rel_distance = q_pos[:,:,None] - k_pos[:,None,:]  # i-j, (batch_size, query_length, key_length)
            rel_distance.clamp_min_(0)  # max{i-j, 0}
            rel_distance = rel_distance.type_as(x)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = rel_distance
        else:
            rel_distance = self.cached_matrix

        self.L_sqrt = self.L_sqrt.to(x.device)
        rel_distance_max, _ = torch.max(rel_distance, dim=-1) # i, (batch_size, query_length)
        rel_distance_max.clamp_max_(self.L_sqrt.square()) # max{L, i}

        self.c = self.c.to(x.device)
        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1)  # (batch_size, query_length, key_length)
        rel_distance_max = torch.log(torch.abs(self.c * rel_distance_max) + 1).unsqueeze(-1)  # (batch_size, query_length, 1)
        normalized_distance = rel_distance / (rel_distance_max + self.eps)  # (batch_size, query_length, key_length)
        position_bias = self.net(normalized_distance.unsqueeze(-1)) # (batch_size, query_length, key_length, num_heads)

        return position_bias.permute([0, 3, 1, 2]) # (batch_size, num_heads, query_length, key_length, num_heads)
