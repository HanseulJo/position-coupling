cd ../..

n_train=30
n_test=200
n_layers=6
n_heads=8
maxpos=203
lr=0.00003

# nopad_plain
for seed in 0 1 2 3; do
for seed_data in 0 1; do
for n_digits in 10 20 30 40; do
    python attention_matrix.py \
        --n_digits $n_digits \
        --compile \
        --title "Attention Pattern of layer ?, head ! \(Position Coupling: Zero-Padding X Reverse X)" \
        --overrides \
            ++best=False \
            device=cuda:2 \
            seed=$seed \
            seed_data=$seed_data \
            group_name=Addition_${n_train}_${n_test} \
            exp_name=coupled_nopad_plain_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_LR${lr} \
            task=addition_coupled \
            task.reverse_input=False \
            task.reverse_output=False \
            task.padding=False \
            task.val_long.n_data=10000 \
            training.batch_size_eval=50;
done
done
done

# nopad_revall
for seed in 0 1 2 3; do
for seed_data in 0 1; do
for n_digits in 10 20 30 40; do
    python attention_matrix.py \
        --n_digits $n_digits \
        --compile \
        --title "Attention Pattern of layer ?, head ! \(Position Coupling: Zero-Padding X Reverse Query&Answer)" \
        --overrides \
            ++best=False \
            device=cuda:2 \
            seed=$seed \
            seed_data=$seed_data \
            group_name=Addition_${n_train}_${n_test} \
            exp_name=coupled_nopad_revall_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_LR${lr} \
            task=addition_coupled \
            task.reverse_input=True \
            task.reverse_output=True \
            task.padding=False \
            task.val_long.n_data=10000 \
            training.batch_size_eval=50;
done
done
done

# nopad_revout
for seed in 0 1 2 3; do
for seed_data in 0 1; do
for n_digits in 10 20 30 40; do
    python attention_matrix.py \
        --n_digits $n_digits \
        --compile \
        --title "Attention Pattern of layer ?, head ! \(Position Coupling: Zero-Padding X Reverse Answer)" \
        --overrides \
            ++best=False \
            device=cuda:2 \
            seed=$seed \
            seed_data=$seed_data \
            group_name=Addition_${n_train}_${n_test} \
            exp_name=coupled_nopad_revout_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_LR${lr} \
            task=addition_coupled \
            task.reverse_input=False \
            task.reverse_output=True \
            task.padding=False \
            task.val_long.n_data=10000 \
            training.batch_size_eval=50;
done
done
done



# pad_plain
for seed in 0 1 2 3; do
for seed_data in 0 1; do
for n_digits in 10 20 30 40; do
    python attention_matrix.py \
        --n_digits $n_digits \
        --compile \
        --title "Attention Pattern of layer ?, head ! \(Position Coupling: Zero-Padding O Reverse X)" \
        --overrides \
            ++best=False \
            device=cuda:2 \
            seed=$seed \
            seed_data=$seed_data \
            group_name=Addition_${n_train}_${n_test} \
            exp_name=coupled_pad_plain_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_LR${lr} \
            task=addition_coupled \
            task.reverse_input=False \
            task.reverse_output=False \
            task.padding=True \
            task.val_long.n_data=10000 \
            training.batch_size_eval=50;
done
done
done

# pad_revall
for seed in 0 1 2 3; do
for seed_data in 0 1; do
for n_digits in 10 20 30 40; do
    python attention_matrix.py \
        --n_digits $n_digits \
        --compile \
        --title "Attention Pattern of layer ?, head ! \(Position Coupling: Zero-Padding O Reverse Query&Answer)" \
        --overrides \
            ++best=False \
            device=cuda:2 \
            seed=$seed \
            seed_data=$seed_data \
            group_name=Addition_${n_train}_${n_test} \
            exp_name=coupled_pad_revall_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_LR${lr} \
            task=addition_coupled \
            task.reverse_input=True \
            task.reverse_output=True \
            task.padding=True \
            task.val_long.n_data=10000 \
            training.batch_size_eval=50;
done
done
done

# pad_revout
for seed in 0 1 2 3; do
for seed_data in 0 1; do
for n_digits in 10 20 30 40; do
    python attention_matrix.py \
        --n_digits $n_digits \
        --compile \
        --title "Attention Pattern of layer ?, head ! \(Position Coupling: Zero-Padding O Reverse Answer)" \
        --overrides \
            ++best=False \
            device=cuda:2 \
            seed=$seed \
            seed_data=$seed_data \
            group_name=Addition_${n_train}_${n_test} \
            exp_name=coupled_pad_revout_maxpos${maxpos}_${n_layers}layers_${n_heads}heads_LR${lr} \
            task=addition_coupled \
            task.reverse_input=False \
            task.reverse_output=True \
            task.padding=True \
            task.val_long.n_data=10000 \
            training.batch_size_eval=50;
done
done
done