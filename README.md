# Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure

Github repository for:
* Hanseul Cho, Jaeyoung Cha, Pranjal Awasthi, Srinadh Bhojanapalli, Anupam Gupta, and Chulhee Yun. "Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure." NeurIPS 2024. 🥳 [arxiv.org/abs/2405.20671](https://arxiv.org/abs/2405.20671)
* Hanseul Cho, Jaeyoung Cha, Srinadh Bhojanapalli, and Chulhee Yun. "Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count." arXiv preprint. [arxiv.org/abs/2410.15787](https://arxiv.org/abs/2410.15787)


## Citations

```bibtex
@inproceedings{cho2024position,
    title={Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure}, 
    author={Hanseul Cho and Jaeyoung Cha and Pranjal Awasthi and Srinadh Bhojanapalli and Anupam Gupta and Chulhee Yun},
    booktitle={Advances in Neural Information Processing Systems},
    volume={38},
    year={2024}
}

@article{cho2024arithmetic,
    title={Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count}, 
    author={Hanseul Cho and Jaeyoung Cha and Srinadh Bhojanapalli and Chulhee Yun},
    journal={arXiv preprint arXiv:2410.15787},
    year={2024},
}
```

## Conda Environment Setting

Minimal environment to run our code base:

```bash
conda env create -f env.yaml
```

## How to run our codes

If you want to train a single model with a single combination of random seeds, you may run `run.py`. Use `--override` to change the model/task/training configurations as you want.

```bash
python run.py \
    --override \
        use_wandb=True \
        group_name="<GroupName>" \
        exp_name="<ExperimentName>" \
        seed=999 \
        seed_data=42 \
        model="CustomT5DecoderOnly" \
        model.position_encoding_type="abs_learned" \
        model.num_layers=6 \
        model.num_heads=8 \
        model.save=True \
        task="addition_coupled" \
        task.max_position=102 \
        task.train.n_data=1000000 \
        training="default" \
        training.batch_size_train=1000 \
        training.batch_size_eval=100 \
        training.n_steps=50000 \
        training.optimizer.lr=0.0001
```

The result will be logged in the `log/` directory. An example of the file structure of the logging directory is as follows:

```
log/
└── <GroupName>
    └── <ExperimentName>
        └── seed999_seedData42
            ├── cfg.json 
            ├── best_<MODEL_NAME>.pt
            ├── last_<MODEL_NAME>.pt
            ├── loss.pdf
            ├── instancewise_accuracy.pdf
            └── tokenwise_accuracy.pdf
```

If you have multiple number of devices (e.g., GPUs), we highly recommend you to run `run_parallel.py` to train the models with exactly the same configuration but with different combinations of random seeds.

```bash
python run_parallel.py \
    --use_wandb \
    --group_name "<GroupName>" \
    --exp_name "<ExperimentName>" \
    --seeds 0 1 2 3 \
    --seeds_data 0 1 \
    --devices 0 1 2 3 \
    --num_exp_per_device 2 \
    --override \
        model="CustomT5DecoderOnly" \
        model.position_encoding_type="abs_learned" \
        model.num_layers=6 \
        model.num_heads=8 \
        model.save=True \
        task="addition_coupled" \
        task.max_position=102 \
        task.train.n_data=1000000 \
        training="default" \
        training.batch_size_train=1000 \
        training.batch_size_eval=100 \
        training.n_steps=50000 \
        training.optimizer.lr=0.0001
```

For more examples of running codes, please check `scripts/` directory.


## Remarks

* Our modeling codes (e.g., `CustomT5DecoderOnly`) are mostly based on the modification by [this repository](https://github.com/McGill-NLP/length-generalization). 
    - Our code basically supports various positional embedding (PE) schemes such as Rotary PE, T5's relative bias, Alibi, Absolute Fixed PE, etc. We also manually implemented [FIRE](https://openreview.net/forum?id=rR03qFesqk). However, they are not tested except for NoPE (`model.position_encoding_type="none"`) and Absolute Learned PE (`model.position_encoding_type="abs_learned"`).
* We use [Hydra](https://hydra.cc) to maintain the configurations.


## File Structure
```
.
├── attention_matrix.py     (only for `CustomT5DecoderOnly` model)
├── env.yaml                (Conda environment)
├── evaluate_model.py       (model evaluation)
├── run.py
├── run_parallel.py
├── configs/
│   ├── config.yaml
│   ├── model/
│   │   ├── CustomT5DecoderOnly.yaml
│   │   └── ... other model configs ...
│   ├── task/
│   │   ├── addition.yaml
│   │   ├── addition_coupled.yaml
│   │   ├── addition_index_hint.yaml
│   │   └── ... other task configs ...
│   └── training/
│       └── default.yaml
├── dataset/ (generated by running code)
├── log/     (generated by running code)
├── scripts/
│   ├── addition/
│   │   ├── run_<METHOD>.sh
│   │   ├── eval_<METHOD>.sh
│   │   └── attn_mtx.sh
│   ├── Nx2multiplication/
│   │   ├── run_<METHOD>.sh
│   │   └── eval_<METHOD>.sh
│   └── ... other folders of script files for other tasks ...
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   └── training_utils.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── arithmetic_dataset.py   (build dataset here)
│   │   ├── common.py               (Parent class `ArithmeticDataset`)
│   │   └── <TASK_NAME>.py          (addition, multiplication, ...)
│   ├── evaluate/
│   │   ├── __init__.py
│   │   └── accuracy.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── build_model.py
│   │   └── modeling/
│   │       ├── custom_gpt2.py
│   │       ├── custom_t5_decoder_only.py   (our main model)
│   │       └── positional_embeddings.py
│   ├── tokenization/
│   │   ├── __init__.py
│   │   └── tokenization.py
│   └── training/
│       ├── __init__.py
│       └── optimization.py
├── vis/  (make it yourself, for visualization)
└── wandb/  (automatically generated when using W&B)
```


