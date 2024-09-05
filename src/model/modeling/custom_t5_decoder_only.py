"""
    Source: https://github.com/McGill-NLP/length-generalization/blob/main/src/models/custom_t5_decoder_only.py
"""
import logging
import math
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, T5Config
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.models.t5.modeling_t5 import (
    T5Stack,
    T5PreTrainedModel,
    T5Block,
    T5LayerNorm,
    T5DenseActDense,
    T5DenseGatedActDense,
    T5LayerFF,
    T5LayerSelfAttention,
    T5Attention,
    T5Model, 
    T5ForConditionalGeneration, 
    T5EncoderModel, 
    T5ForQuestionAnswering,
    T5ClassificationHead
)
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
from tokenizers import Tokenizer

logger = logging.getLogger("app")

from src.model.modeling.positional_embeddings import *


############# Normalization Layer ##############


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


#######################################


class CustomT5Attention(T5Attention):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super(T5Attention, self).__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.d_head = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.factor = config.initializer_factor
        self.tempered_softmax = getattr(config, 'tempered_softmax', False)  # boolean

        if self.tempered_softmax:
            self.tau = torch.nn.Parameter(torch.normal(0., getattr(config, 'tempered_softmax_std', 0.02), (1,)).float())

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.position_encoding_type = getattr(
            config, "position_encoding_type", POSITION_ENCODING_REL_T5_BIAS
        )

        if (self.position_encoding_type == POSITION_ENCODING_REL_T5_BIAS and self.has_relative_attention_bias) \
            or self.position_encoding_type == POSITION_ENCODING_COUPLED_REL_BIAS:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
         
        elif self.position_encoding_type == POSITION_ENCODING_REL_TRANSFORMER_XL:
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_heads, self.d_head))
            nn.init.normal_(
                self.r_r_bias, mean=0.0, std=self.factor * 0.2
            )
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_heads, self.d_head))
            nn.init.normal_(
                self.r_w_bias, mean=0.0, std=self.factor * 0.2
            )
            self.r = nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
            self.r.weight.data.normal_(
                mean=0.0, std=self.factor * (self.d_model**-0.5)
            )
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.clamp_length = 1000

        elif self.position_encoding_type == POSITION_ENCODING_ROTARY:
            self.rotary_dim = getattr(config, "rotary_dim", self.d_head//4)

        elif self.position_encoding_type == POSITION_ENCODING_ROTARY_NEW:
            # We hardcode the rotary dim to 25 percent of the head dim
            self.rotary_dim = self.d_head // 4

        elif self.position_encoding_type == POSITION_ENCODING_FIRE:
            self.d_fire = getattr(config, 'd_fire', 32)
            self.fire_model = Fire(
                self.d_fire, 
                self.n_heads
            )
            for module in self.fire_model.nets:
                nn.init.normal_(module[0].weight, mean=0.0, std=self.factor * 1.0)
                nn.init.normal_(module[2].weight, mean=0.0, std=self.factor * (self.d_fire**-0.5))
                nn.init.normal_(module[4].weight, mean=0.0, std=self.factor * (self.d_fire**-0.5))

        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def _rel_shift(self, x):
        zero_pad_shape = x.size()[:2] + (x.size(2), 1)
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)
        x_padded_shape = x.size()[:2] + (x.size(3) + 1, x.size(2))
        x_padded = x_padded.view(*x_padded_shape)
        x = x_padded[:, :, 1:, :].view_as(x)
        return x

    def compute_bias_fire(self, query_length, key_length, device=None):
        """Compute binned relative position bias for FIRE"""
        context_position = torch.arange(start=1, end=query_length+1, dtype=torch.bfloat16, device=device)[:, None]
        memory_position = torch.arange(start=1, end=key_length+1, dtype=torch.bfloat16, device=device)[None, :]
        relative_position = torch.div(- context_position + memory_position, context_position)  # shape (query_length, key_length)
        values = self.fire_model(relative_position)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        key_value_states=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )

        def shape(states: torch.Tensor):
            """projection"""
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states: torch.Tensor):
            """reshape"""
            return (
                states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key states
        if self.position_encoding_type in [
            POSITION_ENCODING_ROTARY,
            POSITION_ENCODING_ROTARY_NEW,
        ]:
            key_states = shape(self.k(hidden_states))
        else:
            key_states = project(
                hidden_states,
                self.k,
                key_value_states,
                past_key_value[0] if past_key_value is not None else None,
            )
        
        # get value states
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        attention_output_dict = {}

        if self.position_encoding_type == POSITION_ENCODING_REL_T5_BIAS:
            scores = torch.matmul(query_states, key_states.transpose(3, 2))
            attention_output_dict["scores_before"] = scores
            if position_bias is None:
                if not self.has_relative_attention_bias:
                    position_bias = torch.zeros(
                        (1, self.n_heads, real_seq_length, key_length),
                        device=scores.device,
                        dtype=scores.dtype,
                    )
                    if self.gradient_checkpointing and self.training:
                        position_bias.requires_grad = True
                else:
                    position_bias = self.compute_bias(real_seq_length, key_length)

                # if key and values are already calculated
                # we want only the last query position bias
                if past_key_value is not None:
                    position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

                if mask is not None:
                    position_bias = (
                        position_bias + mask
                    )  # (batch_size, n_heads, seq_length, key_length)

            scores += position_bias
        
        elif self.position_encoding_type == POSITION_ENCODING_REL_TRANSFORMER_XL:
            if position_bias is None:
                pos_seq = torch.arange(
                    real_seq_length - 1,
                    -1,
                    -1.0,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                if self.clamp_length > 0:
                    pos_seq = pos_seq.clamp_(max=self.clamp_length)
                position_bias = self.pos_emb(pos_seq)
                position_bias = nn.functional.dropout(
                    position_bias, p=self.dropout, training=self.training
                )

            position_embeds = position_bias  # position embeds: [1, seq_len, d_model]

            r_head_k = self.r(position_embeds)  # [1, seq_len, n_head*d_head]
            r_head_k = r_head_k.view(
                position_embeds.shape[1], self.n_heads, self.d_head
            )  # [seq_len, n_head, d_head]

            rw_head_q = query_states + self.r_w_bias[None, :, None, :]
            AC = torch.einsum("bnqd,bnkd->bnqk", (rw_head_q, key_states))

            rr_head_q = query_states + self.r_r_bias[None, :, None, :]
            BD = torch.einsum("bnid,jnd->bnij", (rr_head_q, r_head_k))
            BD = self._rel_shift(BD)

            scores = AC + BD
            if mask is not None:
                scores += mask

        elif self.position_encoding_type == POSITION_ENCODING_ROTARY:
            r_seq_len = hidden_states.shape[1]
            r_offset = 0

            if past_key_value is not None:
                r_offset = past_key_value[0].shape[2]
                r_seq_len += r_offset

            query_states = query_states.permute(0, 2, 1, 3)
            key_states = key_states.permute(0, 2, 1, 3)

            if self.rotary_dim is not None:

                k_rot = key_states[:, :, :, : self.rotary_dim]
                k_pass = key_states[:, :, :, self.rotary_dim :]

                q_rot = query_states[:, :, :, : self.rotary_dim]
                q_pass = query_states[:, :, :, self.rotary_dim :]

                sincos = fixed_pos_embedding(k_rot, 1, seq_len=r_seq_len)
                k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=r_offset)
                q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=r_offset)

                if output_attentions:
                    scores_pass = torch.matmul(
                        q_pass.permute(0, 2, 1, 3),
                        k_pass.permute(0, 2, 1, 3).transpose(3, 2),
                    )
                    attention_output_dict["scores_pass"] = scores_pass

                    scores_rot = torch.matmul(
                        q_rot.permute(0, 2, 1, 3),
                        k_rot.permute(0, 2, 1, 3).transpose(3, 2),
                    )
                    attention_output_dict["scores_rot"] = scores_rot

                key_states = torch.cat([k_rot, k_pass], dim=-1)
                query_states = torch.cat([q_rot, q_pass], dim=-1)
            else:
                sincos = fixed_pos_embedding(key_states, 1, seq_len=r_seq_len)
                key_states = apply_rotary_pos_emb(key_states, sincos, offset=r_offset)
                query_states = apply_rotary_pos_emb(
                    query_states, sincos, offset=r_offset
                )

            query_states = query_states.permute(0, 2, 1, 3)
            key_states = key_states.permute(0, 2, 1, 3)

            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)

            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
            if mask is not None:
                scores += mask  # (batch_size, n_heads, seq_length, key_length)

        elif self.position_encoding_type == POSITION_ENCODING_ROTARY_NEW:
            r_seq_len = hidden_states.shape[1]
            r_offset = 0

            if past_key_value is not None:
                r_offset = past_key_value[0].shape[2]
                r_seq_len += r_offset

            query_states = query_states.permute(0, 2, 1, 3)
            key_states = key_states.permute(0, 2, 1, 3)

            if self.rotary_dim is not None:
                k_rot = key_states[:, :, :, : self.rotary_dim]
                k_pass = key_states[:, :, :, self.rotary_dim :]

                q_rot = query_states[:, :, :, : self.rotary_dim]
                q_pass = query_states[:, :, :, self.rotary_dim :]

                sincos = position_bias
                # sincos is just vector created by torch.cat([sin, cos], dim=-1)
                # so we can just split it in half
                sin = sincos[:, :, : self.rotary_dim // 2]
                cos = sincos[:, :, self.rotary_dim // 2 :]

                # We don't need to pass offset here, because we already used
                # position_ids to retrieve correct sin and cos vectors
                k_rot = apply_rotary_pos_emb_new(k_rot, (sin, cos))
                q_rot = apply_rotary_pos_emb_new(q_rot, (sin, cos))

                key_states = torch.cat([k_rot, k_pass], dim=-1)
                query_states = torch.cat([q_rot, q_pass], dim=-1)
            else:
                raise ValueError("rotary_dim is None")

            query_states = query_states.permute(0, 2, 1, 3)
            key_states = key_states.permute(0, 2, 1, 3)

            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)

            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
            if mask is not None:
                scores += mask  # (batch_size, n_heads, seq_length, key_length)

        elif self.position_encoding_type == POSITION_ENCODING_ALiBi:
            scores = torch.matmul(query_states, key_states.transpose(3, 2))
            attention_output_dict["scores_before"] = scores

            alibi = position_bias
            alibi = alibi.view(batch_size, self.n_heads, 1, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                alibi = alibi[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                alibi = alibi + mask  # (batch_size, n_heads, seq_length, key_length)

            scores += alibi

        elif self.position_encoding_type == POSITION_ENCODING_FIRE:
            scores = torch.matmul(query_states, key_states.transpose(3, 2))
            if position_bias is None:
                position_bias = self.compute_bias_fire(seq_length, seq_length, scores.device)

            if mask is not None:
                position_bias = position_bias + mask

            scores += position_bias

        elif self.position_encoding_type == POSITION_ENCODING_COUPLED_REL_BIAS:
            scores = torch.matmul(query_states, key_states.transpose(3, 2))
            attention_output_dict["scores_before"] = scores

            relative_position = position_bias.long()
            num_buckets = self.relative_attention_num_buckets
            
            relative_position_bucket = torch.clip(
                relative_position + num_buckets//2,
                torch.tensor(0).to(relative_position.device),
                torch.tensor(num_buckets-1).to(relative_position.device)
            )
            position_bias_rel = self.relative_attention_bias(relative_position_bucket) # shape (..., batchsize, query_length, key_length, num_heads)
            if position_bias_rel.dim() == 5:  # If multi-level position ID comes in
                position_bias_rel = position_bias_rel.sum(0)  # shape (batchsize, query_length, key_length, num_heads)
            position_bias_rel = position_bias_rel.permute([0, 3, 1, 2]) # shape (batchsize, num_heads, query_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias_rel = position_bias_rel[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias_rel += mask # (batch_size, n_heads, seq_length, key_length)
            
            scores += position_bias_rel
        
        else:
            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

            if mask is not None:
                scores += mask  # (batch_size, n_heads, seq_length, key_length)

        if self.tempered_softmax:
            scores[scores != float('-inf')] *= 1. + self.tau * torch.log(torch.tensor(seq_length)).to(device=scores.device)

        attention_output_dict["scores"] = scores

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attention_output_dict["probs"] = attn_weights

        attn_output = unshape(
            torch.matmul(attn_weights, value_states)
        )  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attention_output_dict,)
        return outputs


class CustomT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super(T5LayerSelfAttention, self).__init__()
        self.SelfAttention = CustomT5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        if config.normalization_layer == 't5layernorm':
            norm = T5LayerNorm
        elif config.normalization_layer == 'layernorm':
            norm = nn.LayerNorm
        elif config.normalization_layer == 'rmsnorm':
            norm = RMSNorm
        else:
            raise ValueError(f"{config.normalization_layer} is not implemented normalization layer."
                             "Useable options: ['layernorm', 't5layernorm', 'rmsnorm'].")
        self.layer_norm = norm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm_position = config.layer_norm_position
        if self.layer_norm_position == 'pre_post':
            self.layer_norm_2 = norm(config.d_model, eps=config.layer_norm_epsilon)
        self.alpha = 1.0
        if getattr(config, 'deepnorm', False):
            if config.is_encoder_decoder:
                self.alpha = math.pow(3.0 * config.num_decoder_layers, 0.25)
            else:
                self.alpha = math.pow(2.0 * config.num_decoder_layers, 0.25)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):  
        if self.layer_norm_position in ['pre', 'pre_post']:
            normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            hidden_states if self.layer_norm_position == 'post' else normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = self.alpha * hidden_states + self.dropout(attention_output[0])
        if self.layer_norm_position == 'post':
            hidden_states = self.layer_norm(hidden_states)
        elif self.layer_norm_position == 'pre_post':
            hidden_states = self.layer_norm_2(hidden_states)
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class CustomT5LayerFF(T5LayerFF):
    def __init__(self, config: T5Config):
        super(CustomT5LayerFF, self).__init__(config)
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        if config.normalization_layer == 't5layernorm':
            norm = T5LayerNorm
        elif config.normalization_layer == 'layernorm':
            norm = nn.LayerNorm
        elif config.normalization_layer == 'rmsnorm':
            norm = RMSNorm
        else:
            raise ValueError(f"{config.normalization_layer} is not implemented normalization layer."
                             "Useable options: ['layernorm', 't5layernorm', 'rmsnorm'].")
        self.layer_norm = norm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm_position = config.layer_norm_position
        if self.layer_norm_position == 'pre_post':
            self.layer_norm_2 = norm(config.d_model, eps=config.layer_norm_epsilon)
        self.alpha = 1.0
        if getattr(config, 'deepnorm', False):
            if config.is_encoder_decoder:
                self.alpha = math.pow(3.0 * config.num_decoder_layers, 0.25)
            else:
                self.alpha = math.pow(2.0 * config.num_decoder_layers, 0.25)

    def forward(self, hidden_states):
        if self.layer_norm_position in ['pre', 'pre_post']:
            forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(hidden_states if self.layer_norm_position == 'post' else forwarded_states)
        hidden_states = self.alpha * hidden_states + self.dropout(forwarded_states)
        if self.layer_norm_position == 'post':
            hidden_states = self.layer_norm(hidden_states)
        elif self.layer_norm_position == 'pre_post':
            hidden_states = self.layer_norm_2(hidden_states)
        return hidden_states


class CustomT5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super(T5Block, self).__init__()
        self.is_decoder = config.is_decoder
        assert self.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            CustomT5LayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        )
        self.layer.append(CustomT5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        self_attn_past_key_value = past_key_value

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.empty(
        (target_length, target_length + past_key_values_length),
        dtype=torch.bool,
        device=device,
    )
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(
        batch_size, 1, target_length, target_length + past_key_values_length
    )
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


class CustomT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super(T5Stack, self).__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.position_encoding_type = getattr(
            config, "position_encoding_type", POSITION_ENCODING_NONE
        )

        logger.info(f"position_encoding_type: {self.position_encoding_type}")

        self.block = nn.ModuleList(
            [
                CustomT5Block(config, has_relative_attention_bias=bool(i == 0) and self.position_encoding_type in [
                    POSITION_ENCODING_REL_T5_BIAS,
                    POSITION_ENCODING_REL_TRANSFORMER_XL
                ])
                for i in range(config.num_layers)
            ]
        )
        # self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.position_dim = getattr(config, 'd_positions', None)
        if self.position_encoding_type == POSITION_ENCODING_ABS_LEARNED:
            maxpos = getattr(config, 'n_positions', 2048)
            if self.position_dim is None:
                self.wpe = nn.Embedding(maxpos, config.d_model)
                parent_dir = Path(os.path.dirname(os.path.abspath(__file__)))
                learned_embed_file = parent_dir / "gpt_neo_125m_pos_embed.npy"
                if learned_embed_file.exists():
                    logger.info(
                        "Loading position embedding from {}".format(learned_embed_file)
                    )
                    import numpy as np

                    weight = np.load(str(learned_embed_file))
                    self.wpe.weight.data.copy_(torch.from_numpy(weight))
                    self.wpe.weight.requires_grad = False
                else:
                    self.wpe.weight.data.normal_(
                        mean=0.0, std=config.initializer_factor * 1.0
                    )
            elif self.position_dim >= 1:  # Support for multi-dimensional position ids
                if getattr(config, 'share_pe', False):
                    self.wpe = nn.ModuleList([nn.Embedding(maxpos, config.d_model)] * self.position_dim)
                else:
                    self.wpe = nn.ModuleList([nn.Embedding(maxpos, config.d_model) for _ in range(self.position_dim)])
            

        if self.position_encoding_type == POSITION_ENCODING_ABS_SINUSOID:
            self.wpe = FixedAbsolutePositionalEmbedding(config.d_model)

        if self.position_encoding_type == POSITION_ENCODING_ROTARY_NEW:
            # Rotary dim is X percentage of d_head
            # Right now, we just hardcode X here following:
            # https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/models/gpt_neox/configuration_gpt_neox.py
            rotary_dim = int(config.d_kv * 0.25)
            self.fixed_rotary_embedding = FixedRotaryPositionalEmbedding(
                rotary_dim, max_position=4096
            )

        if self.position_encoding_type in [
            POSITION_ENCODING_ALiBi,
            POSITION_ENCODING_ALiBi_LEARNED,
        ]:
            maxpos = getattr(config, 'n_positions', 2048)
            attn_heads = config.num_heads
            if self.position_encoding_type == POSITION_ENCODING_ALiBi_LEARNED:
                self.learned_logslopes = nn.Parameter(
                    torch.log(torch.Tensor(self.get_slopes(attn_heads)))
                )
            else:
                slopes = torch.Tensor(self.get_slopes(attn_heads))
                alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(
                    maxpos
                ).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
                alibi = alibi.view(attn_heads, 1, maxpos)
                self.register_buffer("alibi", alibi)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def _alibi_prepare_attn_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        past_key_values_length: int,
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n
            )  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n)
            )  # when the number of heads is not a power of 2, we use this workaround.
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self.get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        position_ids=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            assert (
                self.embed_tokens is not None
            ), "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        if self.position_encoding_type in [
            POSITION_ENCODING_ABS_LEARNED,
            POSITION_ENCODING_ABS_SINUSOID,
        ]:  
            ## To support multi-dimensional position ids (of shape (d_positions, batch, seq_len)),
            ## We commented out this part!
            if position_ids is not None and position_ids.dim() <= 2:
                position_ids = position_ids.view(-1, input_shape[-1])

            if past_key_values is None:
                past_length = 0
            else:
                past_length = past_key_values[0][0].size(-2)

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if position_ids is None:
                position_ids = torch.arange(
                    past_length,
                    input_shape[-1] + past_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

            if self.position_dim is None:
                position_embeds = self.wpe(position_ids)
            elif self.position_dim >= 1:
                assert self.position_dim == position_ids.size(0), f"{self.position_dim} != {position_ids.size(0)}"
                position_embeds = sum(pe(pid) for pe, pid in zip(self.wpe, position_ids))
            inputs_embeds += position_embeds

        batch_size, seq_length = input_shape

        # `position_bias` is a just tensor that is passed to all attention layers
        position_bias = None

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            past_key_values[0][0].shape[2] + seq_length
            if past_key_values is not None
            else seq_length
        )

        if use_cache is True:
            assert (
                self.is_decoder
            ), f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(
                inputs_embeds.device
            )

        if self.position_encoding_type in [
                POSITION_ENCODING_ROTARY_NEW,
                POSITION_ENCODING_COUPLED_REL_BIAS
            ]:
            if position_ids is not None and position_ids.dim() <= 2:
                position_ids = position_ids.view(-1, input_shape[-1])

            if past_key_values is None:
                past_length = 0
            else:
                past_length = past_key_values[0][0].size(-2)

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if position_ids is None:
                position_ids = torch.arange(
                    past_length,
                    input_shape[-1] + past_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            
            if self.position_encoding_type == POSITION_ENCODING_ROTARY_NEW:
                sinusoidal_pos = self.fixed_rotary_embedding(position_ids)
                position_bias = sinusoidal_pos
            
            elif self.position_encoding_type == POSITION_ENCODING_COUPLED_REL_BIAS:
                ## Relative PE variant of Position Coupling (Cho et al., 2024) ##
                context_position = position_ids.unsqueeze(-1)  # (..., batchsize, seqlen, 1)
                memory_position = position_ids.unsqueeze(-2)   # (..., batchsize, 1, seqlen)
                relative_position = memory_position - context_position
                position_bias = relative_position  # (..., batchsize, seqlen, seqlen)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, inputs_embeds.device
        )

        if self.position_encoding_type == POSITION_ENCODING_ALiBi:
            num_heads = self.config.num_heads
            alibi = build_alibi_tensor(
                attention_mask, num_heads, dtype=inputs_embeds.dtype
            )
            position_bias = alibi

        if self.position_encoding_type in [POSITION_ENCODING_ALiBi_LEARNED]:
            if not hasattr(self, "alibi"):
                maxpos = 2048
                attn_heads = self.config.num_heads
                slopes = self.learned_logslopes.exp()
                alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(
                    maxpos, device=slopes.device
                ).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
                alibi = alibi.view(attn_heads, 1, maxpos)
            else:
                alibi = self.alibi

            alibi = alibi.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            alibi = alibi[:, :, :, : attention_mask.shape[-1]]
            alibi = alibi.repeat(1, 1, extended_attention_mask.shape[2], 1)
            extended_attention_mask = torch.where(
                extended_attention_mask == 0,
                alibi,
                extended_attention_mask.repeat(1, self.config.num_heads, 1, 1),
            )

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        # position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.block, past_key_values)
        ):
            layer_head_mask = head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    layer_head_mask=layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]

            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state,
                )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (None,)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class CustomDecoderOnlyT5(T5PreTrainedModel):

    _keys_to_ignore_on_load_missing = [
        r"decoder\.embed_tokens\.weight",
        r"encoder",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        position_encoding_type: Optional[str] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ):
        assert config is not None
        config.is_decoder = True
        config.is_encoder_decoder = False
        if position_encoding_type is not None:
            if position_encoding_type == POSITION_ENCODING_ROTARY_RERUN:
                position_encoding_type = POSITION_ENCODING_ROTARY

            if position_encoding_type not in [
                POSITION_ENCODING_ALiBi,
                POSITION_ENCODING_ALiBi_LEARNED,
                POSITION_ENCODING_ABS_LEARNED,
                POSITION_ENCODING_ABS_SINUSOID,
                POSITION_ENCODING_REL_T5_BIAS,
                POSITION_ENCODING_REL_TRANSFORMER_XL,
                POSITION_ENCODING_ROTARY,
                POSITION_ENCODING_ROTARY_NEW,
                POSITION_ENCODING_NONE,
                POSITION_ENCODING_FIRE,
                POSITION_ENCODING_COUPLED_REL_BIAS,
            ]:
                raise ValueError(
                    f"Invalid position_encoding_type: {position_encoding_type}"
                )
            config.position_encoding_type = position_encoding_type

        self.main_input_name = "input_ids"

        super().__init__(config)

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = CustomT5Stack(config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.handle_tokenizer(tokenizer)

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        beta = 1.0
        if getattr(self.config, 'deepnorm', False):
            if self.config.is_encoder_decoder:
                beta = math.pow(12.0 * self.config.num_decoder_layers, -0.25)
            else:
                beta = math.pow(8.0 * self.config.num_decoder_layers, -0.25)

        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(
            module,
            (T5Model, T5ForConditionalGeneration, T5EncoderModel, T5ForQuestionAnswering),
        ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "qa_outputs"):
                module.qa_outputs.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
                module.qa_outputs.bias.data.zero_()
        elif isinstance(module, T5ClassificationHead):
            module.dense.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.dense, "bias") and module.dense.bias is not None:
                module.dense.bias.data.zero_()
            module.out_proj.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.out_proj, "bias") and module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=beta * factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=beta * factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=beta * factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=beta * factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=beta * factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=beta * factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=beta * factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def get_decoder(self):
        return self.decoder

    def handle_tokenizer(self, tokenizer: Optional[Tokenizer] = None):
        if tokenizer is None:
            return

        self.config.eos_token_id = tokenizer.eos_token_id
        self.config.bos_token_id = self.config.eos_token_id

        if (
            len(tokenizer) > self.shared.num_embeddings
            or len(tokenizer) < self.config.vocab_size
        ):
            logger.info(
                f"Resizing num_embeddings to {len(tokenizer)} (Previously, {self.shared.num_embeddings})"
            )
            self.resize_token_embeddings(len(tokenizer))

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.decoder.deparallelize()
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = position_ids[:input_ids.size(-1)]

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            if input_ids is not None:
                input_ids = input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)

        transformer_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            hidden_states = hidden_states * (self.model_dim**-0.5)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Compute loss in fp32 to match with mesh-tf version
            # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )
    

