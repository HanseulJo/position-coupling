import torch
from torch import nn
from transformers import (
    EncoderDecoderConfig,
    EncoderDecoderModel,
)
import src.model as m


CUSTOM_T5_DECODER_ONLY = "CustomT5DecoderOnly"
CUSTOM_GPT2 = "CustomGPT2"
DECODER_BASED = [
    "GPT2",
    CUSTOM_T5_DECODER_ONLY,
    CUSTOM_GPT2,
]
ENCODER_DECODER_BASED = [
    "BertEncoderDecoder",
    "Bart",
    "T5",
]

def _build_encoder_decoder_model_from_scratch(model_cfg, tokenizer, device):
    if "encoder" in model_cfg and "decoder" in model_cfg:
        config_encoder = getattr(m, f"{model_cfg.encoder.model_name}Config")(
            vocab_size=tokenizer.get_vocab_size(),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            forced_eos_token_id=tokenizer.eos_token_id,
            **(model_cfg.encoder)
        )
        config_decoder = getattr(m, f"{model_cfg.decoder.model_name}Config")(
            vocab_size=tokenizer.get_vocab_size(),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,
            forced_eos_token_id=tokenizer.eos_token_id,
            **(model_cfg.decoder)
        )
        config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
        config.vocab_size=tokenizer.get_vocab_size(),
        config.pad_token_id = tokenizer.pad_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.decoder_start_token_id = tokenizer.bos_token_id
        config.forced_eos_token_id = tokenizer.eos_token_id
        config.is_encoder_decoder = True
        model = EncoderDecoderModel(config).to(device)
    else:
        config = getattr(m, f"{model_cfg.model_name}Config")(
            vocab_size=tokenizer.get_vocab_size(),
            pad_token_id = tokenizer.pad_token_id,
            bos_token_id = tokenizer.bos_token_id,
            eos_token_id = tokenizer.eos_token_id,
            decoder_start_token_id = tokenizer.bos_token_id,
            forced_eos_token_id = tokenizer.eos_token_id,
            **model_cfg
        )
        model = getattr(m, f"{model_cfg.model_name}ForConditionalGeneration")(config).to(device)
    return model


def _build_decoder_model_from_scratch(model_cfg, tokenizer, device):
    config = getattr(m, f"{model_cfg.model_name}Config")(
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id = tokenizer.pad_token_id,
        bos_token_id = tokenizer.bos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        decoder_start_token_id = tokenizer.bos_token_id,
        forced_eos_token_id = tokenizer.eos_token_id,
        **model_cfg
    )
    model = getattr(m, f"{model_cfg.model_name}LMHeadModel")(config).to(device)
    return model


def _build_custom_model_from_scratch(model_cfg, tokenizer, device):
    if model_cfg.model_name == CUSTOM_T5_DECODER_ONLY:
        config = m.T5Config(
            vocab_size=tokenizer.get_vocab_size(),
            pad_token_id = tokenizer.pad_token_id,
            bos_token_id = tokenizer.bos_token_id,
            eos_token_id = tokenizer.eos_token_id,
            decoder_start_token_id = tokenizer.bos_token_id,
            forced_eos_token_id = tokenizer.eos_token_id,
            **model_cfg
        )
        model = m.CustomDecoderOnlyT5(
            config,
            position_encoding_type=model_cfg.position_encoding_type
        ).to(device)
    elif model_cfg.model_name == CUSTOM_GPT2:
        config = m.CustomGPT2Config(
            vocab_size=tokenizer.get_vocab_size(),
            bos_token_id = tokenizer.bos_token_id,
            eos_token_id = tokenizer.eos_token_id,
            **model_cfg
        )
        model = m.CustomGPT2LMHeadModel(config).to(device)
    return model


def build_model_from_scratch(cfg, tokenizer, device):
    model_cfg = cfg.model
    if model_cfg.model_name in [ CUSTOM_T5_DECODER_ONLY ]:
        model = _build_custom_model_from_scratch(model_cfg, tokenizer, device)
    elif model_cfg.model_name in ENCODER_DECODER_BASED:
        model = _build_encoder_decoder_model_from_scratch(model_cfg, tokenizer, device)
    elif model_cfg.model_name in DECODER_BASED:
        model = _build_decoder_model_from_scratch(model_cfg, tokenizer, device) 
    else:
        raise ValueError(f"model_name {model_cfg.model_name} is entered")
    return model
        

def build_auxiliary_model(cfg, tokenizer, model):
    model_aux = build_model_from_scratch(cfg, tokenizer)
    for k, _ in model.named_children():
        if k != 'lm_head':
            setattr(model_aux, k, getattr(model, k))
    return model_aux