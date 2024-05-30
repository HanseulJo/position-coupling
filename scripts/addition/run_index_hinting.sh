cd ../..

n_train=30
n_test=100
maxpos=202
n_layers=6
n_heads=8
norm_type=rmsnorm
norm_pos=pre_post
act=gated-gelu
lr=0.0001
wd=0
d_kv=$((512/n_heads))
hide_index_hints=False
bs=400

python run_parallel.py \
    --group_name Addition_${n_train}_${n_test} \
    --exp_name IndexHint_RandomAPE_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_${act}_${norm_pos}_${norm_type}_LR${lr}_WD${wd}_BS${bs}_hidehints${hide_index_hints} \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 0 1 2 3 \
    --num_exp_per_device 1 \
    --overrides \
        model.position_encoding_type=abs_learned \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=$norm_type \
        model.layer_norm_position=$norm_pos \
        model.feed_forward_proj=$act \
        model.d_ff=2048 \
        model.d_kv=$d_kv \
        model.n_positions=2048 \
        model.save=True \
        task=addition_index_hint \
        +task.hide_index_hints=$hide_index_hints \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.max_position=$maxpos \
        task.train.n_data=1000000 \
        task.val.min_n_digits=$n_train \
        task.val.max_n_digits=$n_train \
        task.val.n_data=10000 \
        task.val_long.min_n_digits=$n_test \
        task.val_long.max_n_digits=$n_test \
        task.val_long.n_data=10000 \
        training.batch_size_train=$bs \
        training.batch_size_eval=100 \
        training.n_steps=50000 \
        training.optimizer.lr=$lr \
        training.optimizer.weight_decay=$wd

python run_parallel.py \
    --group_name Addition_${n_train}_${n_test} \
    --exp_name IndexHint_NoPE_${n_layers}layers_${n_heads}heads_${act}_${norm_pos}_${norm_type}_LR${lr}_WD${wd}_BS${bs}_hidehints${hide_index_hints} \
    --seeds 0 1 2 3 \
    --seeds_data 0 \
    --devices 0 1 2 3 \
    --num_exp_per_device 1 \
    --overrides \
        model.position_encoding_type=none \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=$norm_type \
        model.layer_norm_position=$norm_pos \
        model.feed_forward_proj=$act \
        model.d_ff=2048 \
        model.d_kv=$d_kv \
        model.save=True \
        task=addition_index_hint \
        +task.hide_index_hints=$hide_index_hints \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.max_position=$maxpos \
        task.train.n_data=1000000 \
        task.val.min_n_digits=$n_train \
        task.val.max_n_digits=$n_train \
        task.val.n_data=10000 \
        task.val_long.min_n_digits=$n_test \
        task.val_long.max_n_digits=$n_test \
        task.val_long.n_data=10000 \
        training.batch_size_train=$bs \
        training.batch_size_eval=100 \
        training.n_steps=50000 \
        training.optimizer.lr=$lr \
        training.optimizer.weight_decay=$wd
