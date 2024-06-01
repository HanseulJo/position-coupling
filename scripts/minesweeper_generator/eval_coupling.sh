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


for best in True False; do
for share_pe in False True; do
for seed in 0 1 2 3; do
for seed_data in 0 1; do
python evaluate_model.py \
    --min_n_digits 5 \
    --max_n_digits 15 \
    --step 1 \
    --overrides \
        ++best=$best \
        device=cuda:3 \
        group_name=MinesweeperGenerator_${min_n_train}_${max_n_train}_${n_test} \
        exp_name=coupled_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_${act}_${norm_pos}_${norm_type}_LR${lr}_WD${wd}_BS${bs}_shared${share_pe} \
        seed=$seed \
        seed_data=$seed_data \
        ++model.d_positions=2 \
        ++model.share_pe=$share_pe \
        task=minesweeper_generator_coupled \
        task.train.min_n_len=$min_n_train \
        task.train.max_n_len=$max_n_train \
        task.max_position=$maxpos \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=100000 \
        training.batch_size_eval=50
done
done
done
done