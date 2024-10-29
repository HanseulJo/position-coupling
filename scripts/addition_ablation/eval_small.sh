cd ../..

n_train=30
n_test=200
maxpos=203
n_layers=1
n_heads=4



for best in False; do
for seed in 0 1 2 3; do
for seed_data in 0 1; do
python evaluate_model.py \
    --min_n_digits 5 \
    --max_n_digits 200 \
    --step 5 \
    --overrides \
        ++best=$best \
        device=cuda:0 \
        group_name=Addition_${n_train}_${n_test} \
        exp_name=coupled_nopad_plain_maxpos${maxpos}_${n_layers}layers_${n_heads}heads \
        seed=$seed \
        seed_data=$seed_data \
        task=addition_coupled \
        task.reverse_input=False \
        task.reverse_output=False \
        task.padding=False \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.max_position=$maxpos \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=10000 \
        training.batch_size_eval=100
done
done
done

for best in False; do
for seed in 0 1 2 3; do
for seed_data in 0 1; do
python evaluate_model.py \
    --min_n_digits 5 \
    --max_n_digits 200 \
    --step 5 \
    --overrides \
        ++best=$best \
        device=cuda:0 \
        group_name=Addition_${n_train}_${n_test} \
        exp_name=coupled_nopad_revall_maxpos${maxpos}_${n_layers}layers_${n_heads}heads \
        seed=$seed \
        seed_data=$seed_data \
        task=addition_coupled \
        task.reverse_input=True \
        task.reverse_output=True \
        task.padding=False \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.max_position=$maxpos \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=10000 \
        training.batch_size_eval=400
done
done
done

for best in False; do
for seed in 0 1 2 3; do
for seed_data in 0 1; do
python evaluate_model.py \
    --min_n_digits 5 \
    --max_n_digits 200 \
    --step 5 \
    --overrides \
        ++best=$best \
        device=cuda:0 \
        group_name=Addition_${n_train}_${n_test} \
        exp_name=coupled_nopad_revout_maxpos${maxpos}_${n_layers}layers_${n_heads}heads \
        seed=$seed \
        seed_data=$seed_data \
        task=addition_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.padding=False \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.max_position=$maxpos \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=10000 \
        training.batch_size_eval=400
done
done
done


##############


for best in False; do
for seed in 0 1 2 3; do
for seed_data in 0 1; do
python evaluate_model.py \
    --min_n_digits 5 \
    --max_n_digits 200 \
    --step 5 \
    --overrides \
        ++best=$best \
        device=cuda:0 \
        group_name=Addition_${n_train}_${n_test} \
        exp_name=coupled_pad_plain_maxpos${maxpos}_${n_layers}layers_${n_heads}heads \
        seed=$seed \
        seed_data=$seed_data \
        task=addition_coupled \
        task.reverse_input=False \
        task.reverse_output=False \
        task.padding=True \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.max_position=$maxpos \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=10000 \
        training.batch_size_eval=400
done
done
done

for best in False; do
for seed in 0 1 2 3; do
for seed_data in 0 1; do
python evaluate_model.py \
    --min_n_digits 5 \
    --max_n_digits 200 \
    --step 5 \
    --overrides \
        ++best=$best \
        device=cuda:0 \
        group_name=Addition_${n_train}_${n_test} \
        exp_name=coupled_pad_revall_maxpos${maxpos}_${n_layers}layers_${n_heads}heads \
        seed=$seed \
        seed_data=$seed_data \
        task=addition_coupled \
        task.reverse_input=True \
        task.reverse_output=True \
        task.padding=True \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.max_position=$maxpos \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=10000 \
        training.batch_size_eval=400
done
done
done

for best in False; do
for seed in 0 1 2 3; do
for seed_data in 0 1; do
python evaluate_model.py \
    --min_n_digits 5 \
    --max_n_digits 200 \
    --step 5 \
    --overrides \
        ++best=$best \
        device=cuda:0 \
        group_name=Addition_${n_train}_${n_test} \
        exp_name=coupled_pad_revout_maxpos${maxpos}_${n_layers}layers_${n_heads}heads \
        seed=$seed \
        seed_data=$seed_data \
        task=addition_coupled \
        task.reverse_input=False \
        task.reverse_output=True \
        task.padding=True \
        task.train.min_n_digits=1 \
        task.train.max_n_digits=$n_train \
        task.max_position=$maxpos \
        task.train.n_data=1 \
        task.val.n_data=1 \
        task.val_long.n_data=10000 \
        training.batch_size_eval=400
done
done
done