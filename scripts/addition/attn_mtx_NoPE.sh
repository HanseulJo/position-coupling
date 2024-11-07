cd ../..

n_train=30
n_test=200
n_layers=6
n_heads=8

for seed in 0 1 2 3; do
for seed_data in 0 1; do
for n_digits in 10 20 30 40; do
    python attention_matrix.py \
        --n_digits $n_digits \
        --compile \
        --title "Attention Pattern of layer ?, head ! (NoPE)" \
        --overrides \
            ++best=False \
            device=cuda:1 \
            seed=$seed \
            seed_data=$seed_data \
            group_name=Addition_${n_train}_${n_test} \
            exp_name=NoPE_pad_revout_${n_layers}layers_${n_heads}heads \
            task=addition \
            task.reverse_input=False \
            task.reverse_output=True \
            task.padding=True \
            task.val_long.n_data=10000 \
            training.batch_size_eval=50;
done
done
done
# done
# done