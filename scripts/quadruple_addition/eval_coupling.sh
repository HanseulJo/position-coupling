cd ../..

n_train=40
n_test=100
maxpos=301
n_layers=1
n_heads=4
norm_type=rmsnorm
norm_pos=pre_post
act=gated-gelu
lr=0.0001
wd=0
bs=1000


for best in True False; do
for seed in 0 1 2 3; do
for seed_data in 0 1; do
python evaluate_model.py \
    --min_n_digits 5 \
    --max_n_digits 200 \
    --step 5 \
    --overrides \
        ++best=$best \
        device=cuda:0 \
        group_name=QuadrupleAddition_${n_train}_${n_test} \
        exp_name=coupled_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_${act}_${norm_pos}_${norm_type}_LR${lr}_WD${wd}_BS${bs} \
        seed=$seed \
        seed_data=$seed_data \
        task=multiple_addition_coupled \
        task.train.min_n_operands=4 \
        task.train.max_n_operands=4 \
        task.val.min_n_operands=4 \
        task.val.max_n_operands=4 \
        task.val_long.min_n_operands=4 \
        task.val_long.max_n_operands=4 \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.max_position=$maxpos \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=100000 \
        training.batch_size_eval=100
done
done
done
