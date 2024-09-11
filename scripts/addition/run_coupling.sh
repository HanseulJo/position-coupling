cd ../..

n_train=30
n_test_medium=50
n_test=100
maxpos=150
norm_type=rmsnorm
norm_pos=pre_post
act=gated-gelu
lr=0.0001
wd=0
d_model=1024
d_ff=2048
bs=1000


#########
## For Printing Attention Matrices
# n_train=5
# n_test=10
# maxpos=17
# n_layers=1
# n_heads=2
# norm_type=rmsnorm
# norm_pos=pre_post
# act=gated-gelu
# lr=0.0001
# wd=0
# d_kv=$((512/n_heads))
# bs=1000
###########

n_layers=1
n_heads=4
d_kv=$((d_model/n_heads))
python run_parallel.py \
    --use_wandb \
    --group_name Addition_${n_train}_${n_test} \
    --exp_name coupledAbs_pad_revout_${n_layers}layers_${n_heads}heads \
    --seeds 0 1 2 \
    --seeds_data 0 1 \
    --devices 1 \
    --num_exp_per_device 3 \
    --overrides \
        project_name='PositionCouplingRelative' \
        model.position_encoding_type=abs_learned \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=$norm_type \
        model.layer_norm_position=$norm_pos \
        model.feed_forward_proj=$act \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.n_positions=$((maxpos+1)) \
        model.save=True \
        task=addition_coupled \
        task.max_position=$maxpos \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=100000 \
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
        training.batch_size_train=$bs \
        training.batch_size_eval=50 \
        training.n_steps=50000 \
        training.optimizer.lr=$lr \
        training.optimizer.weight_decay=$wd


n_layers=2
n_heads=8
d_kv=$((d_model/n_heads))
python run_parallel.py \
    --use_wandb \
    --group_name Addition_${n_train}_${n_test} \
    --exp_name coupledAbs_pad_revout_${n_layers}layers_${n_heads}heads \
    --seeds 0 1 2 \
    --seeds_data 0 1 \
    --devices 1 \
    --num_exp_per_device 2 \
    --overrides \
        project_name='PositionCouplingRelative' \
        model.position_encoding_type=abs_learned \
        model.num_layers=$n_layers \
        model.num_heads=$n_heads \
        model.normalization_layer=$norm_type \
        model.layer_norm_position=$norm_pos \
        model.feed_forward_proj=$act \
        model.d_model=$d_model \
        model.d_ff=$d_ff \
        model.d_kv=$d_kv \
        model.n_positions=$((maxpos+1)) \
        model.save=True \
        task=addition_coupled \
        task.max_position=$maxpos \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.train.n_data=100000 \
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
        training.batch_size_train=$bs \
        training.batch_size_eval=50 \
        training.n_steps=50000 \
        training.optimizer.lr=$lr \
        training.optimizer.weight_decay=$wd