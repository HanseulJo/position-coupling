cd ../..

n_train=20
n_test=50
n_layers=6
n_heads=8

python evaluate_model_parallel.py \
    --runner_name evaluate_model \
    --group_name Parity_${n_train}_${n_test} \
    --exp_name NoPE_${n_layers}layers_${n_heads}heads \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 4 5 6 7 0 1  \
    --num_exp_per_device 2 \
    --min_n_digits 5 \
    --max_n_digits 100 \
    --step_digits 5 \
    --compile \
    --overrides \
        ++best=False \
        task=parity \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=10000 \
        training.batch_size_eval=100

python evaluate_model_parallel.py \
    --runner_name evaluate_model \
    --group_name Parity_${n_train}_${n_test} \
    --exp_name FIRE_${n_layers}layers_${n_heads}heads \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 4 5 6 7 0 1 \
    --num_exp_per_device 2 \
    --min_n_digits 5 \
    --max_n_digits 100 \
    --step_digits 5 \
    --compile \
    --overrides \
        ++best=False \
        task=parity \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=10000 \
        training.batch_size_eval=100


python evaluate_model_parallel.py \
    --runner_name evaluate_model \
    --group_name Parity_${n_train}_${n_test} \
    --exp_name RoPE_${n_layers}layers_${n_heads}heads \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 4 5 6 7 0 1 \
    --num_exp_per_device 2 \
    --min_n_digits 5 \
    --max_n_digits 100 \
    --step_digits 5 \
    --compile \
    --overrides \
        ++best=False \
        ++model.rotary_dim=$d_kv \
        ++model.rotary_base=10000 \
        task=parity \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=10000 \
        training.batch_size_eval=100
