cd ../..

n_train=10
n_test=20
m_train=10
m_test=20
n_layers=6
n_heads=8
lr=0.00003
wd=0
d_model=1024
n_data=500000
bs=400
clip=1
d_ff=2048
d_kv=$((d_model/n_heads)) 

for best in False True; do
for seed in 0; do
for seed_data in 0; do
python evaluate_model_multiple_addition.py \
    --min_n_digits 1 \
    --max_n_digits 15 \
    --min_n_operands 2 \
    --max_n_operands 15 \
    --step_digits 1 \
    --step_operands 1 \
    --pad_offset 0 \
    --compile \
    --overrides \
        ++seed=$seed \
        ++seed_data=$seed_data \
        ++best=$best \
        device='cuda:1' \
        group_name=MultipleAdditionScratchpad_di${n_train}_${n_test}_op${m_train}_${m_test} \
        exp_name=coupled_pad_revall_${n_layers}L${n_heads}H${d_model}dim_Data${n_data}BS${bs}LR${lr}WD${wd} \
        ++model.d_positions=2 \
        ++model.share_pe=False \
        task=multiple_addition_scratchpad_coupled \
        task.reverse_input=True \
        task.reverse_output=True \
        task.reverse_output_order=False \
        task.padding=True \
        ++task.scratchpad_start_with_zeros=True \
        ++task.train.pad_offset=0 \
        ++task.val.pad_offset=0 \
        ++task.val_many_digits.pad_offset=0 \
        ++task.val_many_operands.pad_offset=0 \
        ++task.val_long.pad_offset=0 \
        task.val_long.n_data=1000 \
        training.batch_size_eval=10
done
done
done



