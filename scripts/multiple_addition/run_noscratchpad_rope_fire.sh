cd ../..

n_train=10
n_test=20
m_train=10
m_test=20

n_layers=6
n_heads=8

lr=0.00003
wd=0
d_model=1024
d_ff=2048
d_kv=$((d_model/n_heads)) 

# n_data=10000
n_data=500000
bs=400

layernorm=rmsnorm
norm_pos=pre_post
act=gated-gelu


## RoPE, no scratchpad ##
python run_parallel.py \
    --use_wandb \
    --group_name MultipleAddition_di${n_train}_${n_test}_op${m_train}_${m_test} \
    --exp_name RoPE_noCoT_${n_layers}L${n_heads}H${d_model}dim_Data${n_data}BS${bs}LR${lr}WD${wd} \
    --seeds 0 1 \
    --seeds_data 0 1 \
    --devices 2 1 0 \
    --num_exp_per_device 2 \
    --overrides \
        model.position_encoding_type=rotary_new \
        ++model.rotary_dim=$d_kv \
        ++model.rotary_base=10000 \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=$layernorm \
        model.layer_norm_position=$norm_pos \
        model.feed_forward_proj=$act \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.save=True \
        task=multiple_addition \
        task.reverse_input=False \
        task.reverse_output=True \
        task.reverse_output_order=False \
        task.padding=True \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.min_n_operands=2 \
        task.train.max_n_operands=$m_train \
        task.train.n_data=$n_data \
        task.train.sampling_method_n_digits=partially_uniform \
        task.train.threshold_partially_uniform=0.5 \
        task.val.min_n_digits=1 \
        task.val.max_n_digits=$n_train \
        task.val.min_n_operands=2 \
        task.val.max_n_operands=$m_train \
        task.val.n_data=1000 \
        task.val_many_digits.min_n_digits=$((n_train+1)) \
        task.val_many_digits.max_n_digits=$n_test \
        task.val_many_digits.min_n_operands=2 \
        task.val_many_digits.max_n_operands=$m_train \
        task.val_many_digits.n_data=1000 \
        task.val_many_operands.min_n_digits=1 \
        task.val_many_operands.max_n_digits=$n_train \
        task.val_many_operands.min_n_operands=$((m_train+1)) \
        task.val_many_operands.max_n_operands=$m_test \
        task.val_many_operands.n_data=1000 \
        task.val_long.min_n_digits=$((n_train+1)) \
        task.val_long.max_n_digits=$n_test \
        task.val_long.min_n_operands=$((m_train+1)) \
        task.val_long.max_n_operands=$m_test \
        task.val_long.n_data=1000 \
        training.batch_size_train=$bs \
        training.batch_size_eval=50 \
        training.n_steps=50000 \
        training.optimizer.lr=$lr \
        training.optimizer.weight_decay=$wd


## FIRE, no scratchpad ##
python run_parallel.py \
    --use_wandb \
    --group_name MultipleAddition_di${n_train}_${n_test}_op${m_train}_${m_test} \
    --exp_name FIRE_noCoT_${n_layers}L${n_heads}H${d_model}dim_Data${n_data}BS${bs}LR${lr}WD${wd} \
    --seeds 0 1 \
    --seeds_data 0 1 \
    --devices 2 1 0 \
    --num_exp_per_device 2 \
    --overrides \
        model.position_encoding_type=fire \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=$layernorm \
        model.layer_norm_position=$norm_pos \
        model.feed_forward_proj=$act \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.save=True \
        task=multiple_addition \
        task.reverse_input=False \
        task.reverse_output=True \
        task.reverse_output_order=False \
        task.padding=True \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.min_n_operands=2 \
        task.train.max_n_operands=$m_train \
        task.train.n_data=$n_data \
        task.train.sampling_method_n_digits=partially_uniform \
        task.train.threshold_partially_uniform=0.5 \
        task.val.min_n_digits=1 \
        task.val.max_n_digits=$n_train \
        task.val.min_n_operands=2 \
        task.val.max_n_operands=$m_train \
        task.val.n_data=1000 \
        task.val_many_digits.min_n_digits=$((n_train+1)) \
        task.val_many_digits.max_n_digits=$n_test \
        task.val_many_digits.min_n_operands=2 \
        task.val_many_digits.max_n_operands=$m_train \
        task.val_many_digits.n_data=1000 \
        task.val_many_operands.min_n_digits=1 \
        task.val_many_operands.max_n_digits=$n_train \
        task.val_many_operands.min_n_operands=$((m_train+1)) \
        task.val_many_operands.max_n_operands=$m_test \
        task.val_many_operands.n_data=1000 \
        task.val_long.min_n_digits=$((n_train+1)) \
        task.val_long.max_n_digits=$n_test \
        task.val_long.min_n_operands=$((m_train+1)) \
        task.val_long.max_n_operands=$m_test \
        task.val_long.n_data=1000 \
        training.batch_size_train=$bs \
        training.batch_size_eval=50 \
        training.n_steps=50000 \
        training.optimizer.lr=$lr \
        training.optimizer.weight_decay=$wd




