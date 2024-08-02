cd ../..

n_train=30
n_test=100
maxpos=202
n_layers=1
n_heads=4
d_model=512
d_ff=$((d_model*4))
d_kv=$((d_model/n_heads))


python run_parallel.py \
    --group_name Addition_${n_train}_${n_test} \
    --exp_name coupled_pad_plain_maxpos${maxpos}_${n_layers}layers_${n_heads}heads \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 5 \
    --num_exp_per_device 8 \
    --overrides \
        model.position_encoding_type=abs_learned \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=rmsnorm \
        model.layer_norm_position=pre_post \
        model.feed_forward_proj=gated-gelu \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.n_positions=$((maxpos+1)) \
        model.save=True \
        task=addition_coupled \
        task.reverse_input=False \
        task.reverse_output=False \
        task.padding=True \
        task.max_position=$maxpos \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=100000 \
        task.val.min_n_digits=$n_train \
        task.val.max_n_digits=$n_train \
        task.val.n_data=1000 \
        task.val_long.min_n_digits=$n_test \
        task.val_long.max_n_digits=$n_test \
        task.val_long.n_data=1000 \
        training.batch_size_train=100 \
        training.batch_size_eval=50 \
        training.n_steps=50000 \
        training.optimizer.lr=0.0001 \
        training.optimizer.weight_decay=0


n_layers=6
n_heads=8
d_model=1024
d_ff=$((d_model*4))
d_kv=$((d_model/n_heads))


python run_parallel.py \
    --group_name Addition_${n_train}_${n_test} \
    --exp_name coupled_pad_plain_maxpos${maxpos}_${n_layers}layers_${n_heads}heads \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 5 \
    --num_exp_per_device 8 \
    --overrides \
        model.position_encoding_type=abs_learned \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=rmsnorm \
        model.layer_norm_position=pre_post \
        model.feed_forward_proj=gated-gelu \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.n_positions=$((maxpos+1)) \
        model.save=True \
        task=addition_coupled \
        task.reverse_input=False \
        task.reverse_output=False \
        task.padding=True \
        task.max_position=$maxpos \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=100000 \
        task.val.min_n_digits=$n_train \
        task.val.max_n_digits=$n_train \
        task.val.n_data=1000 \
        task.val_long.min_n_digits=$n_test \
        task.val_long.max_n_digits=$n_test \
        task.val_long.n_data=1000 \
        training.batch_size_train=100 \
        training.batch_size_eval=50 \
        training.n_steps=50000 \
        training.optimizer.lr=0.0001 \
        training.optimizer.weight_decay=0