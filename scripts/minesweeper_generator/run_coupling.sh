cd ../..

min_n_train=5
max_n_train=9
n_test=12
maxpos=16
n_layers=4
n_heads=8
norm_type=rmsnorm
norm_pos=pre_post
act=gated-gelu
lr=0.0001
wd=0
d_kv=$((512/n_heads))
bs=800

share_pe=False
python run_parallel.py \
    --group_name MinesweeperGenerator_${min_n_train}_${max_n_train}_${n_test} \
    --exp_name coupled_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_${act}_${norm_pos}_${norm_type}_LR${lr}_WD${wd}_BS${bs}_shared${share_pe} \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 4 5 6 7 \
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
        model.n_positions=$((maxpos+1)) \
        ++model.d_positions=2 \
        ++model.share_pe=$share_pe \
        model.save=True \
        task=minesweeper_generator_coupled \
        task.max_position=$maxpos \
        task.train.min_n_len=$min_n_train \
        task.train.max_n_len=$max_n_train \
        task.train.n_data=1000000 \
        task.val.min_n_len=$max_n_train \
        task.val.max_n_len=$max_n_train \
        task.val.n_data=10000 \
        task.val_long.min_n_len=$n_test \
        task.val_long.max_n_len=$n_test \
        task.val_long.n_data=10000 \
        training.batch_size_train=$bs \
        training.batch_size_eval=50 \
        training.n_steps=100000 \
        training.optimizer.lr=$lr \
        training.optimizer.weight_decay=$wd

share_pe=True
python run_parallel.py \
    --group_name MinesweeperGenerator_${min_n_train}_${max_n_train}_${n_test} \
    --exp_name coupled_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_${act}_${norm_pos}_${norm_type}_LR${lr}_WD${wd}_BS${bs}_shared${share_pe} \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 4 5 6 7 \
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
        model.n_positions=$((maxpos+1)) \
        ++model.d_positions=2 \
        ++model.share_pe=$share_pe \
        model.save=True \
        task=minesweeper_generator_coupled \
        task.max_position=$maxpos \
        task.train.min_n_len=$min_n_train \
        task.train.max_n_len=$max_n_train \
        task.train.n_data=1000000 \
        task.val.min_n_len=$max_n_train \
        task.val.max_n_len=$max_n_train \
        task.val.n_data=10000 \
        task.val_long.min_n_len=$n_test \
        task.val_long.max_n_len=$n_test \
        task.val_long.n_data=10000 \
        training.batch_size_train=$bs \
        training.batch_size_eval=50 \
        training.n_steps=100000 \
        training.optimizer.lr=$lr \
        training.optimizer.weight_decay=$wd