cd ../..

n_train=10
n_test=15
m_train=8
m_test=12
n_layers=6
n_heads=8
lr=0.0001
wd=0
d_model=1024
d_ff=2048
d_kv=$((d_model/n_heads))
n_data=100000
bs=250


python run_parallel.py \
    --use_wandb \
    --group_name MultipleAdditionScratchpad_di${n_train}_${n_test}_op${m_train}_${m_test} \
    --exp_name NoPE_pad_revout_maxpos_di${maxpos_d}_op${maxpos_o}_${n_layers}layers_${n_heads}head_Data${n_data} \
    --seeds 0 1 2 \
    --seeds_data 0 1  \
    --devices 7 \
    --num_exp_per_device 1 \
    --overrides \
        project_name='PositionCoupling with Scratchpad' \
        model.position_encoding_type=none \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=rmsnorm \
        model.layer_norm_position=pre_post \
        model.feed_forward_proj=gated-gelu \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.save=True \
        task=multiple_addition_scratchpad\
        task.reverse_input=False \
        task.reverse_output=True \
        task.reverse_output_order=False \
        task.padding=True \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.min_n_operands=2 \
        task.train.max_n_operands=$m_train \
        task.train.n_data=$n_data \
        task.val.min_n_digits=1 \
        task.val.max_n_digits=$n_train \
        task.val.min_n_operands=2 \
        task.val.max_n_operands=$m_train \
        task.val.n_data=10000 \
        task.val_many_digits.min_n_digits=$((n_train+1)) \
        task.val_many_digits.max_n_digits=$n_test \
        task.val_many_digits.min_n_operands=2 \
        task.val_many_digits.max_n_operands=$m_train \
        task.val_many_digits.n_data=10000 \
        task.val_many_operands.min_n_digits=1 \
        task.val_many_operands.max_n_digits=$n_train \
        task.val_many_operands.min_n_operands=$((m_train+1)) \
        task.val_many_operands.max_n_operands=$m_test \
        task.val_many_operands.n_data=10000 \
        task.val_long.min_n_digits=$((n_train+1)) \
        task.val_long.max_n_digits=$n_test \
        task.val_long.min_n_operands=$((m_train+1)) \
        task.val_long.max_n_operands=$m_test \
        task.val_long.n_data=10000 \
        training.batch_size_train=$bs \
        training.batch_size_eval=50 \
        training.n_steps=50000 \
        training.optimizer.lr=$lr \
        training.optimizer.weight_decay=$wd
