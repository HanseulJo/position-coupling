cd ../..


for n_digits in 11; do
for n_operands in 11; do
for seed in 0 1; do
for seed_data in 0 1; do
    python attention_matrix_multiple_addition.py \
        --n_digits $n_digits \
        --n_operands $n_operands \
        --compile \
        --title "Attention Pattern of layer ?, head !\(NoPE + Scratchpad)" \
        --overrides \
            ++best=False \
            device=cuda:3 \
            seed=$seed \
            seed_data=$seed_data \
            group_name=MultipleAdditionScratchpad_di10_20_op10_20 \
            exp_name=NoPE_6L8H1024dim_Data500000BS400LR0.00003WD0 \
            task=multiple_addition_scratchpad_coupled \
            task.reverse_input=False \
            task.reverse_output=True \
            task.reverse_output_order=False \
            task.padding=True \
            task.val_long.n_data=1000 \
            ++task.scratchpad_start_with_zeros=True \
            training.batch_size_eval=50;
done
done
done
done
# done