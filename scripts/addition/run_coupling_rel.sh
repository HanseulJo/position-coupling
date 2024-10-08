cd ../..

n_train=30
n_test_medium=50
n_test=100

d_model=1024
d_ff=2048



n_layers=1
n_heads=4
d_kv=$((d_model/n_heads))
python run_parallel.py \
    --use_wandb \
    --group_name Addition_${n_train}_${n_test} \
    --exp_name coupledRel_pad_revout_${n_layers}layers_${n_heads}heads_testLogN \
    --seeds 0 1 2 \
    --seeds_data 0 1 \
    --devices 2 \
    --num_exp_per_device 3 \
    --overrides \
        project_name='PositionCouplingRelative' \
        model.position_encoding_type=coupled_relative_bias \
        model.relative_attention_num_buckets=7 \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=rmsnorm \
        model.layer_norm_position=pre_post \
        model.feed_forward_proj=gated-gelu \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.n_positions=null \
        model.save=True \
        task=addition_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.padding=True \
        task.max_position=null \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=100000 \
        task.train.randomize=False \
        task.val.min_n_digits=1 \
        task.val.max_n_digits=$n_train \
        task.val.n_data=10000 \
        +task.val_medium.dataset_cls=AdditionDatasetWithCoupledPositions \
        +task.val_medium.min_n_digits=$((n_train+1)) \
        +task.val_medium.max_n_digits=$n_test_medium \
        +task.val_medium.n_data=10000 \
        +task.val_medium.randomize=False \
        +task.val_medium.hard_carry=False \
        task.val_long.min_n_digits=$((n_test_medium+1)) \
        task.val_long.max_n_digits=$n_test \
        task.val_long.n_data=10000 \
        training.batch_size_train=1000 \
        training.batch_size_eval=50 \
        training.n_steps=50000 \
        training.optimizer.lr=0.0001 \
        training.optimizer.weight_decay=0


n_layers=2
n_heads=8
d_kv=$((d_model/n_heads))
python run_parallel.py \
    --use_wandb \
    --group_name Addition_${n_train}_${n_test} \
    --exp_name coupledRel_pad_revout_${n_layers}layers_${n_heads}heads_testLogN \
    --seeds 0 1 2 \
    --seeds_data 0 1 \
    --devices 2 \
    --num_exp_per_device 2 \
    --overrides \
        project_name='PositionCouplingRelative' \
        model.position_encoding_type=coupled_relative_bias \
        model.relative_attention_num_buckets=7 \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=rmsnorm \
        model.layer_norm_position=pre_post \
        model.feed_forward_proj=gated-gelu \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.n_positions=null \
        model.save=True \
        task=addition_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.padding=True \
        task.max_position=null \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=100000 \
        task.train.randomize=False \
        task.val.min_n_digits=1 \
        task.val.max_n_digits=$n_train \
        task.val.n_data=10000 \
        +task.val_medium.dataset_cls=AdditionDatasetWithCoupledPositions \
        +task.val_medium.min_n_digits=$((n_train+1)) \
        +task.val_medium.max_n_digits=$n_test_medium \
        +task.val_medium.n_data=10000 \
        +task.val_medium.randomize=False \
        +task.val_medium.hard_carry=False \
        task.val_long.min_n_digits=$((n_test_medium+1)) \
        task.val_long.max_n_digits=$n_test \
        task.val_long.n_data=10000 \
        training.batch_size_train=1000 \
        training.batch_size_eval=50 \
        training.n_steps=50000 \
        training.optimizer.lr=0.0001 \
        training.optimizer.weight_decay=0


n_layers=4
python run_parallel.py \
    --use_wandb \
    --group_name Addition_${n_train}_${n_test} \
    --exp_name coupledRel_pad_revout_${n_layers}layers_${n_heads}heads_testLogN \
    --seeds 0 1 2 \
    --seeds_data 0 1 \
    --devices 1 2 \
    --num_exp_per_device 1 \
    --overrides \
        project_name='PositionCouplingRelative' \
        model.position_encoding_type=coupled_relative_bias \
        model.relative_attention_num_buckets=7 \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=rmsnorm \
        model.layer_norm_position=pre_post \
        model.feed_forward_proj=gated-gelu \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.n_positions=null \
        model.save=True \
        task=addition_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.padding=True \
        task.max_position=null \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=100000 \
        task.train.randomize=False \
        task.val.min_n_digits=1 \
        task.val.max_n_digits=$n_train \
        task.val.n_data=10000 \
        +task.val_medium.dataset_cls=AdditionDatasetWithCoupledPositions \
        +task.val_medium.min_n_digits=$((n_train+1)) \
        +task.val_medium.max_n_digits=$n_test_medium \
        +task.val_medium.n_data=10000 \
        +task.val_medium.randomize=False \
        +task.val_medium.hard_carry=False \
        task.val_long.min_n_digits=$((n_test_medium+1)) \
        task.val_long.max_n_digits=$n_test \
        task.val_long.n_data=10000 \
        training.batch_size_train=1000 \
        training.batch_size_eval=50 \
        training.n_steps=50000 \
        training.optimizer.lr=0.0001 \
        training.optimizer.weight_decay=0


n_layers=6
python run_parallel.py \
    --use_wandb \
    --group_name Addition_${n_train}_${n_test} \
    --exp_name coupledRel_pad_revout_${n_layers}layers_${n_heads}heads_testLogN \
    --seeds 0 1 2 \
    --seeds_data 0 1 \
    --devices 1 2 \
    --num_exp_per_device 1 \
    --overrides \
        project_name='PositionCouplingRelative' \
        model.position_encoding_type=coupled_relative_bias \
        model.relative_attention_num_buckets=7 \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=rmsnorm \
        model.layer_norm_position=pre_post \
        model.feed_forward_proj=gated-gelu \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.n_positions=null \
        model.save=True \
        task=addition_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.padding=True \
        task.max_position=null \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=100000 \
        task.train.randomize=False \
        task.val.min_n_digits=1 \
        task.val.max_n_digits=$n_train \
        task.val.n_data=10000 \
        +task.val_medium.dataset_cls=AdditionDatasetWithCoupledPositions \
        +task.val_medium.min_n_digits=$((n_train+1)) \
        +task.val_medium.max_n_digits=$n_test_medium \
        +task.val_medium.n_data=10000 \
        +task.val_medium.randomize=False \
        +task.val_medium.hard_carry=False \
        task.val_long.min_n_digits=$((n_test_medium+1)) \
        task.val_long.max_n_digits=$n_test \
        task.val_long.n_data=10000 \
        training.batch_size_train=500 \
        training.batch_size_eval=50 \
        training.n_steps=50000 \
        training.optimizer.lr=0.0001 \
        training.optimizer.weight_decay=0