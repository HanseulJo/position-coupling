cd ../../..

python run_parallel.py \
    --use_wandb \
    --group_name Addition \
    --exp_name CustomT5DecoderOnly_PEnone \
    --seeds 0 \
    --devices 0 \
    --num_exp_per_device 1 \
    --overrides \
        model=CustomT5DecoderOnly \
        model.position_encoding_type=none \