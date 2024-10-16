cd ../..

n_train=20
n_test=50
n_layers=6
n_heads=8

for best in False True; do
for seed in 2 3 ; do
for seed_data in 0 1; do
python evaluate_model.py \
    --min_n_digits 5 \
    --max_n_digits 100 \
    --step 5 \
    --overrides \
        ++best=$best \
        device=cuda:7 \
        group_name=Parity_${n_train}_${n_test} \
        exp_name=NoPE_plainscratchpad_${n_layers}layers_${n_heads}heads \
        seed=$seed \
        seed_data=$seed_data \
        task=parity_scratchpad \
        task.reversed_scratchpad=False \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=1000 \
        training.batch_size_eval=50
done
done
done