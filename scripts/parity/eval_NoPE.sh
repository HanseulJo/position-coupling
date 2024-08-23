cd ../..

n_train=20
n_test=50
n_layers=6
n_heads=8

for best in Falsed True; do
for seed in 0 1 2 3; do
for seed_data in 0 1; do
python evaluate_model.py \
    --min_n_digits 5 \
    --max_n_digits 50 \
    --step 5 \
    --overrides \
        ++best=$best \
        device=cuda:0 \
        group_name=Parity_${n_train}_${n_test} \
        exp_name=NoPE_${n_layers}layers_${n_heads}heads \
        seed=$seed \
        seed_data=$seed_data \
        task=parity \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=10000 \
        training.batch_size_eval=100
done
done
done