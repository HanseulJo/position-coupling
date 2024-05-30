cd ../../..

python run_parallel.py \
    --use_wandb \
    --group_name Addition \
    --exp_name BertEncDec_relativeK \
    --seeds 0 \
    --devices 0 \
    --num_exp_per_device 1 \
    --overrides \
        model=BertEncoderDecoder \
        model.encoder.position_embedding_type=relative_key \
        model.decoder.position_embedding_type=relative_key \

