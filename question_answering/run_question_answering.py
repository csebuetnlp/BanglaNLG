import logging
import os
import glob
import random
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import numpy as np

import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    MBartTokenizer,
    MBartTokenizerFast,
    MBartForConditionalGeneration,
    AlbertTokenizer,
    AlbertTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from normalizer import normalize

from utils import (
    QADatasetReader,
    find_all_indices,
    QuestionAnsweringTrainer
)

EXT2CONFIG = {
    "json": (QADatasetReader, {})
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
   
    dataset_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Path to the directory containing the data files. (.json)"
            "File datatypes will be identified with their prefix names as follows: "
            "`train`- Training file(s) e.g. `train.json`/ `train_part1.json` etc. "
            "`validation`- Evaluation file(s) e.g. `validation.json`/ `validation_part1.json` etc. "
            "`test`- Test file(s) e.g. `test.json`/ `test_part1.json` etc. "
            "All files for must have the same extension."
        }
    )

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a json file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a json file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a json file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text."}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    
    do_normalize: Optional[bool] = field(default=True, metadata={"help": "Normalize text before feeding to the model."})
    unicode_norm: Optional[str] = field(default="NFKC", metadata={"help": "Type of unicode normalization"})
    lang: Optional[str] = field(default=None, metadata={"help": "Optional language id for bart models"})
    

    def __post_init__(self):
        if self.train_file is not None and self.validation_file is not None:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["json"], "`train_file` should be a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension as `train_file`."


@dataclass
class ModelArguments:
    
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


def main():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    has_ext = lambda path: len(os.path.basename(path).split(".")) > 1
    get_ext = lambda path: os.path.basename(path).split(".")[-1]

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )

    elif data_args.dataset_dir is not None:
        data_files = {}
        all_files = glob.glob(
            os.path.join(
                data_args.dataset_dir,
                "*"
            )
        )
        all_exts = [get_ext(k) for k in all_files if has_ext(k)]
        if not all_exts:
            raise ValueError("The `dataset_dir` doesnt have any valid file.")
            
        selected_ext = max(set(all_exts), key=all_exts.count)
        for search_prefix in ["train", "validation", "test"]:
            found_files = glob.glob(
                os.path.join(
                    data_args.dataset_dir,
                    search_prefix + "*" + selected_ext
                )
            )
            if not found_files:
                continue

            data_files[search_prefix] = found_files

        dataset_configs = EXT2CONFIG[selected_ext]
        raw_datasets = dataset_configs[0](
            data_files, 
            **dataset_configs[1]
        ).read()
        
    else:
        data_files = {
            "train": data_args.train_file, 
            "validation": data_args.validation_file,
            "test": data_args.test_file
        }

        data_files = {k: v for k, v in data_files.items() if v is not None}
        
        if not data_files:
            raise ValueError("No valid input file found.")

        selected_ext = get_ext(list(data_files.values())[0])
        dataset_configs = EXT2CONFIG[selected_ext]
        raw_datasets = dataset_configs[0](
            data_files, 
            **dataset_configs[1]
        ).read()

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )

    tokenizer_kwargs = {"add_prefix_space": True} if config.model_type in {"gpt2", "roberta"} else {}   
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
        **tokenizer_kwargs
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # whether this model is indicbart or its derivative
    is_indicbart = False
    if isinstance(model, MBartForConditionalGeneration) and isinstance(tokenizer, AlbertTokenizer):
        is_indicbart = True
        from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

        code2script = {f"<2{k}>": k for k in ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']}
        bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
        eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
        pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
        
        tokenizer.do_lower_case = False
        tokenizer.keep_accents = True
        
        model.config.pad_token_id = pad_id
        model.config.bos_token_id = bos_id 
        model.config.eos_token_id = eos_id
        

    model.resize_token_embeddings(len(tokenizer))

    if data_args.lang is not None:
        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang
    
        if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)
        elif isinstance(tokenizer, AlbertTokenizer):
            model.config.decoder_start_token_id = tokenizer._convert_token_to_id_with_added_voc(tokenizer.tgt_lang)



    
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]


    if data_args.do_normalize:
        normalization_kwargs = {
            "unicode_norm": data_args.unicode_norm,
        }
        required_column_names = [
            question_column_name,
            context_column_name,
            answer_column_name
        ]

        def normalize_example(example):
            required_row_values = [example[k] for k in required_column_names if k in example]
            question, context = required_row_values[:2]
            example[question_column_name] = normalize(question, **normalization_kwargs)
            example[context_column_name] = normalize(context, **normalization_kwargs)

            if len(required_row_values) == 3:
                answer = required_row_values[2]
                for i, ans in enumerate(answer["text"]):
                    prev_position = answer["answer_start"][i]
                    answer["text"][i] = normalize(ans, **normalization_kwargs)

                    replace_index = -1
                    for j, pos in enumerate(find_all_indices(ans, context)):
                        replace_index = j
                        if pos == prev_position:
                            break

                    if replace_index != -1:
                        index_iterator = find_all_indices(
                            answer["text"][i],
                            example[context_column_name]
                        )
                        for j, pos in enumerate(index_iterator):
                            if j == replace_index:
                                answer["answer_start"][i] = pos
                                assert answer["text"][i] == example[context_column_name][pos: pos + len(answer["text"][i])]
                                break

                example[answer_column_name] = answer

            return example

        raw_datasets = raw_datasets.map(
            normalize_example,
            desc="Running normalization on dataset",
            load_from_cache_file=not data_args.overwrite_cache
        )
    
    pad_on_right = tokenizer.padding_side == "right"
    max_source_length = min(data_args.max_source_length, tokenizer.model_max_length)

    def prepare_features(examples):
        tokenizer_kwargs = {
            "max_length": max_source_length, 
            "padding": False,
            "truncation": "only_second" if pad_on_right else "only_first",
        }

        questions = examples[question_column_name]
        contexts = examples[context_column_name]
        answers = []
        
        for ans in examples[answer_column_name]:
            ans = "" if not ans["text"] else ans["text"][0]
            answers.append(ans)
        
        if is_indicbart:
            if tokenizer.src_lang in code2script:
                questions = [UnicodeIndicTransliterator.transliterate(k, code2script[tokenizer.src_lang], "hi") + f" {tokenizer.eos_token} "
                                for k in questions]
                contexts = [UnicodeIndicTransliterator.transliterate(k, code2script[tokenizer.src_lang], "hi")
                                for k in contexts]
                answers = [UnicodeIndicTransliterator.transliterate(k, code2script[tokenizer.src_lang], "hi")
                                for k in answers]

            tokenizer_kwargs.update({"add_special_tokens": False})
    

        tokenized_examples = tokenizer(
            [prefix + k for k in questions] if pad_on_right else [prefix + k for k in contexts],
            contexts if pad_on_right else questions,
            **tokenizer_kwargs
        )

        if is_indicbart:
            new_features = []
            max_len = max(len(k) for k in tokenized_examples["input_ids"])

            for feature in tokenized_examples["input_ids"]:
                new_features.append(
                    feature + 
                    [eos_id, tokenizer._convert_token_to_id_with_added_voc(tokenizer.src_lang)] +
                    [tokenizer.pad_token_id] * (max_len - len(feature))
                )
            
            tokenized_examples["input_ids"] = np.array(new_features)
            tokenized_examples["attention_mask"] = (tokenized_examples["input_ids"] != tokenizer.pad_token_id).astype(int)
        
        tokenizer_kwargs.update({"max_length": data_args.max_target_length, "truncation": True})

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                answers,
                max_length=data_args.max_target_length,
            )
        
        if is_indicbart:
            new_features = []
            max_len = max(len(k) for k in labels["input_ids"])

            for feature in labels["input_ids"]:
                new_features.append(
                    feature + 
                    [eos_id, tokenizer._convert_token_to_id_with_added_voc(tokenizer.tgt_lang)] +
                    [tokenizer.pad_token_id] * (max_len - len(feature))
                )
            
            labels["input_ids"] = np.array(new_features)

        tokenized_examples["labels"] = labels["input_ids"]
        return tokenized_examples

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                prepare_features,
                batched=True,
                batch_size=1 if is_indicbart else training_args.train_batch_size,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_features,
                batched=True,
                batch_size=1 if is_indicbart else training_args.train_batch_size,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_features,
                batched=True,
                batch_size=1 if is_indicbart else training_args.train_batch_size,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def post_processing_function(examples, predictions, stage="eval"):
        all_predictions = {}
        example_index_to_id = {i: k for i, k in enumerate(examples["id"])}
        output_dir=training_args.output_dir
        
        logger.setLevel(log_level)
        logger.info(f"Post-processing {len(examples)} example predictions")

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        for i, pred in tqdm(enumerate(decoded_preds)):
            all_predictions[example_index_to_id[i]] = pred

        if output_dir is not None:
            assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

            prediction_file = os.path.join(
                output_dir, "predictions.json" if stage is None else f"{stage}_predictions.json"
            )
            
            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")
            
        
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in all_predictions.items()
        ]
    
        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad_v2")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
    
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=data_args.max_target_length, num_beams=data_args.num_beams)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples, max_length=data_args.max_target_length, num_beams=data_args.num_beams)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
