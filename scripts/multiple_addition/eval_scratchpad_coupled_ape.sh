cd ../..

n_train=13
n_test=20
m_train=13
m_test=20
n_layers=6
n_heads=8
lr=0.00003
wd=0
d_model=1024
n_data=500000
bs=200
clip=1
d_ff=2048
d_kv=$((d_model/n_heads)) 

for best in False True; do
for seed in 0 1; do
for seed_data in 0 1; do
python evaluate_model_multiple_addition.py \
    --min_n_digits 1 \
    --max_n_digits 30 \
    --min_n_operands 2 \
    --max_n_operands 30 \
    --step_digits 1 \
    --step_operands 1 \
    --pad_offset 0 \
    --compile \
    --overrides \
        ++seed=$seed \
        ++seed_data=$seed_data \
        ++best=$best \
        device='cuda:2' \
        group_name=MultipleAdditionScratchpad_di${n_train}_${n_test}_op${m_train}_${m_test} \
        exp_name=coupled_${n_layers}L${n_heads}H${d_model}dim_Data${n_data}BS${bs}LR${lr}WD${wd} \
        ++model.rotary_dim=d_kv \
        ++model.rotary_base=10000 \
        ++model.d_positions=2 \
        ++model.share_pe=False \
        task=multiple_addition_scratchpad_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.reverse_output_order=False \
        task.padding=True \
        ++task.scratchpad_start_with_zeros=True \
        ++task.val_long.pad_offset=0 \
        task.val_long.n_data=1000 \
        training.batch_size_eval=10
done
done
done

################################################
################################################

n_train=7
n_test=20
m_train=7
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
for seed in 0 1; do
for seed_data in 0 1; do
python evaluate_model_multiple_addition.py \
    --min_n_digits 1 \
    --max_n_digits 30 \
    --min_n_operands 2 \
    --max_n_operands 30 \
    --step_digits 1 \
    --step_operands 1 \
    --overrides \
        ++seed=$seed \
        ++seed_data=$seed_data \
        ++best=$best \
        device='cuda:2' \
        group_name=MultipleAdditionScratchpad_di${n_train}_${n_test}_op${m_train}_${m_test} \
        exp_name=coupled_${n_layers}L${n_heads}H${d_model}dim_Data${n_data}BS${bs}LR${lr}WD${wd} \
        ++model.rotary_dim=d_kv \
        ++model.rotary_base=10000 \
        ++model.d_positions=2 \
        ++model.share_pe=False \
        task=multiple_addition_scratchpad_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.reverse_output_order=False \
        task.padding=True \
        ++task.scratchpad_start_with_zeros=True \
        ++task.val_long.pad_offset=0 \
        task.val_long.n_data=1000 \
        training.batch_size_eval=5
done
done
done



