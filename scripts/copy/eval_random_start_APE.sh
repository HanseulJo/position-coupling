cd ../..

n_train=40
n_test=100
maxpos=601
n_layers=6
n_heads=8
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
    --max_n_digits 300 \
    --step 5 \
    --overrides \
        ++best=$best \
        device=cuda:0 \
        group_name=Copy_${n_train}_${n_test} \
        exp_name=RandomAPE_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_${act}_${norm_pos}_${norm_type}_LR${lr}_WD${wd}_BS${bs} \
        seed=$seed \
        seed_data=$seed_data \
        task=copy_coupled \
        ++task.vanilla=True \
        task.max_position=$maxpos \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=100000 \
        training.batch_size_eval=50
done
done
done
