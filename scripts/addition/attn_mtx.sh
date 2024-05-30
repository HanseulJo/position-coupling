cd ..

n_train=5
n_layers=1
n_heads=2
norm_type=rmsnorm
norm_pos=pre_post
act=gated-gelu
lr=0.00005
wd=0

d_kv=$((512/n_heads))

maxpos=17
n_test=$((maxpos-2))
for seed in 2; do
for seed_data in 1; do
for n_digits in 6 7; do
    python attention_matrix.py \
        --n_digits $n_digits \
        --overrides \
            +best=True \
            device=cuda:0 \
            seed=$seed \
            seed_data=$seed_data \
            group_name=Addition_${n_train}_${n_test} \
            exp_name=coupled_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_${act}_${norm_pos}_${norm_type}_LR${lr}_WD${wd} \
            task=addition_coupled \
            task.max_position=$maxpos \
            task.train.n_data=1 \
            task.train.min_n_digits=1 \
            task.train.max_n_digits=$n_train \
            task.val.n_data=1 \
            task.val_long.min_n_digits=$n_train \
            task.val_long.max_n_digits=$n_train \
            task.val_long.n_data=10000 \
            task.val_long.min_n_digits=$n_test \
            task.val_long.max_n_digits=$n_test \
            model.position_encoding_type=abs_learned \
            model.num_layers=$n_layers \
            model.num_heads=$n_heads \
            model.normalization_layer=$norm_type \
            model.layer_norm_position=$norm_pos \
            model.feed_forward_proj=$act \
            model.d_ff=2048 \
            model.d_kv=$d_kv \
            model.n_positions=128 \
            training.batch_size_eval=100;
done
done
done
# done
# done