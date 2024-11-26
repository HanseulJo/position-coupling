cd ../..

n_train=10
n_test=20
m_train=10
m_test=20

n_data=500000
bs=400
lr=0.00003
wd=0


d_model=1024
d_ff=2048
n_layers=1
n_heads=16
d_kv=64

python evaluate_model_parallel.py \
    --runner_name evaluate_model_multiple_addition \
    --group_name MultipleAdditionScratchpad_di${n_train}_${n_test}_op${m_train}_${m_test} \
    --exp_name coupled_${n_layers}L${n_heads}H${d_model}D${d_kv}DpH_Data${n_data}BS${bs}LR${lr}WD${wd} \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 0 1 2 3 4 5 6 7 \
    --num_exp_per_device 1 \
    --min_n_digits 1 \
    --max_n_digits 30 \
    --min_n_operands 2 \
    --max_n_operands 30 \
    --step_digits 1 \
    --step_operands 1 \
    --compile \
    --overrides \
        ++best=False \
        ++model.d_positions=2 \
        ++model.share_pe=False \
        task=multiple_addition_scratchpad_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.reverse_output_order=False \
        task.padding=True \
        task.val_long.n_data=1000 \
        training.batch_size_eval=10


d_model=64
d_ff=$((d_model*4))
n_layers=1
n_heads=4
d_kv=$((d_model/n_heads)) 

python evaluate_model_parallel.py \
    --runner_name evaluate_model_multiple_addition \
    --group_name MultipleAdditionScratchpad_di${n_train}_${n_test}_op${m_train}_${m_test} \
    --exp_name coupled_${n_layers}L${n_heads}H${d_model}D${d_kv}DpH_Data${n_data}BS${bs}LR${lr}WD${wd} \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 0 1 2 3 4 5 6 7 \
    --num_exp_per_device 1 \
    --min_n_digits 1 \
    --max_n_digits 30 \
    --min_n_operands 2 \
    --max_n_operands 30 \
    --step_digits 1 \
    --step_operands 1 \
    --compile \
    --overrides \
        ++best=False \
        ++model.d_positions=2 \
        ++model.share_pe=False \
        task=multiple_addition_scratchpad_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.reverse_output_order=False \
        task.padding=True \
        task.val_long.n_data=1000 \
        training.batch_size_eval=10



d_model=256
d_ff=$((d_model*4))
n_layers=1
n_heads=4
d_kv=$((d_model/n_heads)) 

python evaluate_model_parallel.py \
    --runner_name evaluate_model_multiple_addition \
    --group_name MultipleAdditionScratchpad_di${n_train}_${n_test}_op${m_train}_${m_test} \
    --exp_name coupled_${n_layers}L${n_heads}H${d_model}D${d_kv}DpH_Data${n_data}BS${bs}LR${lr}WD${wd} \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 0 1 2 3 4 5 6 7 \
    --num_exp_per_device 1 \
    --min_n_digits 1 \
    --max_n_digits 30 \
    --min_n_operands 2 \
    --max_n_operands 30 \
    --step_digits 1 \
    --step_operands 1 \
    --compile \
    --overrides \
        ++best=False \
        ++model.d_positions=2 \
        ++model.share_pe=False \
        task=multiple_addition_scratchpad_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.reverse_output_order=False \
        task.padding=True \
        task.val_long.n_data=1000 \
        training.batch_size_eval=10


d_model=1024
d_ff=2048
n_layers=1
n_heads=4
d_kv=128

python evaluate_model_parallel.py \
    --runner_name evaluate_model_multiple_addition \
    --group_name MultipleAdditionScratchpad_di${n_train}_${n_test}_op${m_train}_${m_test} \
    --exp_name coupled_${n_layers}L${n_heads}H${d_model}D${d_kv}DpH_Data${n_data}BS${bs}LR${lr}WD${wd} \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 0 1 2 3 4 5 6 7 \
    --num_exp_per_device 1 \
    --min_n_digits 1 \
    --max_n_digits 30 \
    --min_n_operands 2 \
    --max_n_operands 30 \
    --step_digits 1 \
    --step_operands 1 \
    --compile \
    --overrides \
        ++best=False \
        ++model.d_positions=2 \
        ++model.share_pe=False \
        task=multiple_addition_scratchpad_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.reverse_output_order=False \
        task.padding=True \
        task.val_long.n_data=1000 \
        training.batch_size_eval=10



# Warning: only seed 3
d_model=1024
d_ff=2048
n_layers=1
n_heads=8
d_kv=128

python evaluate_model_parallel.py \
    --runner_name evaluate_model_multiple_addition \
    --group_name MultipleAdditionScratchpad_di${n_train}_${n_test}_op${m_train}_${m_test} \
    --exp_name coupled_${n_layers}L${n_heads}H${d_model}dim_Data${n_data}BS${bs}LR${lr}WD${wd}_PartialFullLengthTraining \
    --seeds 3 \
    --seeds_data 0 1 \
    --devices 6 7 \
    --num_exp_per_device 1 \
    --min_n_digits 1 \
    --max_n_digits 30 \
    --min_n_operands 2 \
    --max_n_operands 30 \
    --step_digits 1 \
    --step_operands 1 \
    --compile \
    --overrides \
        ++best=False \
        ++model.d_positions=2 \
        ++model.share_pe=False \
        task=multiple_addition_scratchpad_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.reverse_output_order=False \
        task.padding=True \
        task.val_long.n_data=1000 \
        training.batch_size_eval=10

# Warning: only seed 3
d_model=1024
d_ff=2048
n_layers=1
n_heads=4
d_kv=256

python evaluate_model_parallel.py \
    --runner_name evaluate_model_multiple_addition \
    --group_name MultipleAdditionScratchpad_di${n_train}_${n_test}_op${m_train}_${m_test} \
    --exp_name coupled_${n_layers}L${n_heads}H${d_model}dim_Data${n_data}BS${bs}LR${lr}WD${wd}_PartialFullLengthTraining \
    --seeds 3 \
    --seeds_data 0 1 \
    --devices 6 7 \
    --num_exp_per_device 1 \
    --min_n_digits 1 \
    --max_n_digits 30 \
    --min_n_operands 2 \
    --max_n_operands 30 \
    --step_digits 1 \
    --step_operands 1 \
    --compile \
    --overrides \
        ++best=False \
        ++model.d_positions=2 \
        ++model.share_pe=False \
        task=multiple_addition_scratchpad_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.reverse_output_order=False \
        task.padding=True \
        task.val_long.n_data=1000 \
        training.batch_size_eval=10


# RoPE scratchpad

d_model=1024
n_layers=6
n_heads=8

python evaluate_model_parallel.py \
    --runner_name evaluate_model_multiple_addition \
    --group_name MultipleAdditionScratchpad_di${n_train}_${n_test}_op${m_train}_${m_test} \
    --exp_name RoPE_${n_layers}L${n_heads}H${d_model}dim_Data${n_data}BS${bs}LR${lr}WD${wd} \
    --seeds 0 1 \
    --seeds_data 0 1 \
    --devices 0 1 2 3 \
    --num_exp_per_device 1 \
    --min_n_digits 1 \
    --max_n_digits 30 \
    --min_n_operands 2 \
    --max_n_operands 30 \
    --step_digits 1 \
    --step_operands 1 \
    --compile \
    --overrides \
        ++best=False \
        ++model.rotary_dim=$d_kv \
        ++model.rotary_base=10000 \
        ++model.final_norm=layernorm \
        task=multiple_addition_scratchpad \
        task.reverse_input=False \
        task.reverse_output=True \
        task.reverse_output_order=False \
        task.padding=True \
        task.val_long.n_data=1000 \
        training.batch_size_eval=10