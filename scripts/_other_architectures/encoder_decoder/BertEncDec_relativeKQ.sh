cd ../../..

python run_parallel.py \
    --group_name Addition \
    --exp_name BertEncDec_relativeKQ \
    --seeds 0 \
    --devices 0 \
    --num_exp_per_device 1 \
    --overrides \
        model=BertEncoderDecoder \
        model.encoder.position_embedding_type=relative_key_query \
        model.decoder.position_embedding_type=relative_key_query \


