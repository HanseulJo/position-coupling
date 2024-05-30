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
hide_index_hints=False
bs=400


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
        group_name=Addition_${n_train}_${n_test} \
        exp_name=IndexHint_RandomAPE_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_${act}_${norm_pos}_${norm_type}_LR${lr}_WD${wd}_BS${bs}_hidehints${hide_index_hints} \
        seed=$seed \
        seed_data=$seed_data \
        task=addition_index_hint \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.max_position=$maxpos \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=100000 \
        training.batch_size_eval=20
done
done
done

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
        group_name=Addition_${n_train}_${n_test} \
        exp_name=IndexHint_NoPE_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_${act}_${norm_pos}_${norm_type}_LR${lr}_WD${wd}_BS${bs}_hidehints${hide_index_hints} \
        seed=$seed \
        seed_data=$seed_data \
        task=addition_index_hint \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.max_position=$maxpos \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=100000 \
        training.batch_size_eval=20
done
done
done