cd ../..

n_train=40
n_test=100
maxpos=1023
n_layers=6
n_heads=8
norm_type=rmsnorm
norm_pos=pre_post
act=gated-gelu
lr=0.0001
wd=0
bs=800


for best in True False; do
for seed in 0 1 2 3; do
for seed_data in 0 1; do
python evaluate_model.py \
    --min_n_digits 5 \
    --max_n_digits 100 \
    --step 5 \
    --overrides \
        ++best=$best \
        device=cuda:0 \
        group_name=TripleAddition_${n_train}_${n_test} \
        exp_name=RandomAPE_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_${act}_${norm_pos}_${norm_type}_LR${lr}_WD${wd}_BS${bs} \
        seed=$seed \
        seed_data=$seed_data \
        task=multiple_addition_coupled \
        ++task.vanilla=True \
        task.train.min_n_operands=3 \
        task.train.max_n_operands=3 \
        task.val.min_n_operands=3 \
        task.val.max_n_operands=3 \
        task.val_long.min_n_operands=3 \
        task.val_long.max_n_operands=3 \
        task.max_position=$maxpos \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=100000 \
        training.batch_size_eval=100
done
done
done
