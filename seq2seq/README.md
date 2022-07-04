## Data format

For local files, the finetuning script supports the following input file formats: `csv`, `tsv` and `jsonl`(one json per line). By default, the script expects the following column names (for `tsv`, `csv`) / key names (for `jsonl`):

* `source` - For input text
* `target` - For output text 

You can specify custom key / column names using the flags `--source_key <key_name>`, `--target_key <key_name>` to `run_seq2seq.py`. To view sample input files, see the files **[here](sample_inputs/).**

## Training & Evaluation

To see list of all available options, do `python run_seq2seq.py -h`. There are three ways to provide input data files to the script:

* with flag `--dataset_dir <path>` where `<path>` points to the directory containing files with prefix `train`, `validation` and `test`.
* with flags `--train_file <path>` / `--train_file <path>` / `--validation_file <path>` / `--test_file <path>`.
* with a dataset from [Huggingface Datasets]() Library, usng the keys `--dataset_name <name>` and  `dataset_config_name <name>` (optional)

For the following commands, we are going to use the `--dataset_dir <path>` to provide input files.


### Finetuning
For finetuning and inference on the test set using the best model during validation (on single GPU), a minimal example is as follows:

```bash
$ python ./run_seq2seq.py \
    --model_name_or_path "csebuetnlp/banglat5" \
    --dataset_dir "sample_inputs/" \
    --output_dir "outputs/" \
    --learning_rate=5e-4 \
    --warmup_steps 5000 \
    --label_smoothing_factor 0.1 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.1 \
    --lr_scheduler_type "linear"  \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --max_source_length 256 \
    --max_target_length 256 \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --greater_is_better true --load_best_model_at_end \
    --metric_for_best_model sacrebleu --evaluation_metric sacrebleu \
    --num_train_epochs=20 \ 
    --do_train --do_eval --do_predict \
    --predict_with_generate
```
For a detailed example with multigpu execution, refer to **[trainer.sh](trainer.sh).**


### Evaluation
* To calculate metrics on test set / inference on raw data, use the following snippet:

```bash
$ python ./run_seq2seq.py \
    --model_name_or_path <path/to/trained/model> \
    --dataset_dir "sample_inputs/" \
    --output_dir "outputs/" \
    --per_device_eval_batch_size=8 \
    --overwrite_output_dir \
    --evaluation_metric sacrebleu \
    --do_predict --predict_with_generate
```
