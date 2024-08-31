cd ../..

n_train=10
n_test=15
m_train=5
m_test=7
maxpos_d=25
maxpos_o=10
n_layers=6
n_heads=8
lr=0.0001
wd=0.01
d_model=256
d_ff=1024
d_kv=$((d_model/n_heads))
# n_data=100000
bs=800


for n_data in 100000 1000000 10000000 20000; do
python run_parallel.py \
    --group_name MultiplicationScratchpad_N${n_train}_${n_test}_M${m_train}_${m_test} \
    --exp_name coupled_pad_revout_maxpos_di${maxpos_d}_op${maxpos_o}_${n_layers}layers_${n_heads}heads_Data${n_data} \
    --seeds 0 1 2 \
    --seeds_data 0 1 \
    --devices 4 5 \
    --num_exp_per_device 1 \
    --overrides \
        project_name='PositionCoupling with Scratchpad' \
        model.position_encoding_type=abs_learned \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=rmsnorm \
        model.layer_norm_position=pre_post \
        model.feed_forward_proj=gated-gelu \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.n_positions=$((maxpos_d+1)) \
        ++model.d_positions=3 \
        ++model.share_pe=False \
        model.save=True \
        task=multiplication_scratchpad_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.padding=True \
        task.max_position_digits=$maxpos_d \
        task.max_position_operands=$maxpos_o \
        task.train.min_n_digits_1=1 \
        task.train.max_n_digits_1=$n_train \
        task.train.min_n_digits_2=1 \
        task.train.max_n_digits_2=$m_train \
        task.train.n_data=$n_data \
        task.val.min_n_digits_1=1 \
        task.val.max_n_digits_1=$n_train \
        task.val.min_n_digits_2=1 \
        task.val.max_n_digits_2=$m_train \
        task.val.n_data=10000 \
        task.val_long_first.min_n_digits_1=$((n_train+1)) \
        task.val_long_first.max_n_digits_1=$n_test \
        task.val_long_first.min_n_digits_2=1 \
        task.val_long_first.max_n_digits_2=$m_train \
        task.val_long_first.n_data=10000 \
        task.val_long_second.min_n_digits_1=1 \
        task.val_long_second.max_n_digits_1=$n_train \
        task.val_long_second.min_n_digits_2=$((m_train+1)) \
        task.val_long_second.max_n_digits_2=$m_test \
        task.val_long_second.n_data=10000 \
        task.val_long.min_n_digits_1=$((n_train+1)) \
        task.val_long.max_n_digits_1=$n_test \
        task.val_long.min_n_digits_2=$((m_train+1)) \
        task.val_long.max_n_digits_2=$m_test \
        task.val_long.n_data=10000 \
        training.batch_size_train=$bs \
        training.batch_size_eval=100 \
        training.n_steps=50000 \
        training.optimizer.lr=$lr \
        training.optimizer.weight_decay=$wd
done