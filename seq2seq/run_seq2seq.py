import logging
import os
import sys
import glob
import json
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk
import numpy as np
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer, scoring
from datasets import load_dataset, load_metric
from datasets.io.json import JsonDatasetReader
from datasets.io.csv import CsvDatasetReader

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
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

EXT2CONFIG = {
    "csv" : (CsvDatasetReader, {}),
    "tsv" : (CsvDatasetReader, {"sep": "\t"}),
    "jsonl": (JsonDatasetReader, {}),
    "json": (JsonDatasetReader, {})
}


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    dataset_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Path to the directory containing the data files. (.csv / .tsv / .jsonl)"
            "File datatypes will be identified with their prefix names as follows: "
            "`train`- Training file(s) e.g. `train.csv`/ `train_part1.csv` etc. "
            "`validation`- Evaluation file(s) e.g. `validation.csv`/ `validation_part1.csv` etc. "
            "`test`- Test file(s) e.g. `test.csv`/ `test_part1.csv` etc. "
            "All files for must have the same extension."
        }
    )
    
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
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
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv / tsv / jsonl file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv / tsv / jsonl file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv / tsv / jsonl file containing the test data."})
    do_normalize: Optional[bool] = field(default=False, metadata={"help": "Normalize text before feeding to the model."})
    unicode_norm: Optional[str] = field(default="NFKC", metadata={"help": "Type of unicode normalization"})
    remove_punct: Optional[bool] = field(
        default=False, metadata={
            "help": "Remove punctuation during normalization. To replace with custom token / selective replacement you should "
            "use this repo (https://github.com/abhik1505040/normalizer) before feeding the data to the script."
    })
    remove_emoji: Optional[bool] = field(
        default=False, metadata={
            "help": "Remove emojis during normalization. To replace with custom token / selective replacement you should "
            "use this repo (https://github.com/abhik1505040/normalizer) before feeding the data to the script."
    })
    remove_urls: Optional[bool] = field(
        default=False, metadata={
            "help": "Remove urls during normalization. To replace with custom token / selective replacement you should "
            "use this repo (https://github.com/abhik1505040/normalizer) before feeding the data to the script."
    })
    source_key: Optional[str] = field(
        default="source", metadata={"help": "Key / column name in the input file corresponding to the source data"}
    )
    target_key: Optional[str] = field(
        default="target", metadata={"help": "Key / column name in the input file corresponding to the target data"}
    )

    source_lang: Optional[str] = field(default=None, metadata={"help": "Source language id."})
    target_lang: Optional[str] = field(default=None, metadata={"help": "Target language id."})

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
    val_max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    
    num_beams: Optional[int] = field(
        default=5,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text."}
    )
    evaluation_metric: Optional[str] = field(
        default="rouge",
        metadata={
            "help": "Evaluation metric",
            "choices": ["rouge", "sacrebleu"]
        }    
    )
    rouge_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "Target language for rouge",
        }    
    )

    def __post_init__(self):
        if self.train_file is not None and self.validation_file is not None:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "jsonl", "tsv", "json"], "`train_file` should be a csv / tsv / jsonl file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension csv / tsv / jsonl as `train_file`."


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

    # Log on each process the small summary:
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
        cache_dir=model_args.cache_dir,     
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # whether this model is indicbart or its derivative
    is_indicbart = False
    # whether this model is the unified_script variant of IndicBART
    is_unified = False

    if isinstance(model, MBartForConditionalGeneration) and isinstance(tokenizer, AlbertTokenizer):
        is_indicbart = True
        from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
        import unicodedata
        from collections import Counter

        def get_token_family(token):
            names = Counter([unicodedata.name(c, 'UNKNOWN').split()[0] for c in token])
            return names.most_common(1)[0][0]

        family2count = Counter([get_token_family(t) for t in tokenizer.get_vocab()])
        # enumerating most probable families to allow for moderate vocab change
        ss_requred_unicode_families = ["BENGALI", "TAMIL", "MALAYALAM", "TELUGU", "GURMUKHI", "KANNADA", "GUJARATI", "ORIYA"]
        required_unicode_tokens = sum(family2count.get(k, 0) for k in ss_requred_unicode_families)

        if family2count.get("DEVANAGARI", 0) > required_unicode_tokens:
            is_unified = True
        
        logger.info(f"IndicBART variant: {'US' if is_unified else 'SS'}")

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

    if data_args.source_lang is not None and data_args.target_lang is not None:
        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang
    
        if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)
        elif isinstance(tokenizer, AlbertTokenizer):
            model.config.decoder_start_token_id = tokenizer._convert_token_to_id_with_added_voc(tokenizer.tgt_lang)


    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    for data_type, ds in raw_datasets.items():
        assert data_args.source_key in ds.features, f"Input files doesnt have the `{data_args.source_key}` key"
        if data_type != "test":
            assert data_args.target_key in ds.features, f"Input files doesnt have the `{data_args.target_key}` key"
        
        ignored_columns = set(ds.column_names) - set([data_args.source_key, data_args.target_key])
        raw_datasets[data_type] = ds.remove_columns(ignored_columns)

    max_target_length = data_args.max_target_length
    
    def preprocess_function(examples):
        normalization_kwargs = {
            "unicode_norm": data_args.unicode_norm,
            "punct_replacement": " " if data_args.remove_punct else None,
            "url_replacement": " " if data_args.remove_urls else None,
            "emoji_replacement": " " if data_args.remove_emoji else None
        }
        
        
        inputs = [normalize(ex, **normalization_kwargs) if data_args.do_normalize else ex 
                    for ex in examples[data_args.source_key]]
        inputs = [prefix + inp for inp in inputs]

        tokenizer_kwargs = {
            "max_length": data_args.max_source_length, 
            "padding": False,
            "truncation": True,
            "return_tensors": "np"
        }
        
        if is_indicbart:
            if is_unified and tokenizer.src_lang in code2script:
                inputs = [UnicodeIndicTransliterator.transliterate(k, code2script[tokenizer.src_lang], "hi")
                            for k in inputs]

            tokenizer_kwargs.update({"add_special_tokens": False})
        
        model_inputs = tokenizer(inputs, **tokenizer_kwargs)

        if is_indicbart:
            model_inputs["input_ids"] = np.concatenate(
                (   
                    model_inputs["input_ids"], 
                    np.array([[eos_id, tokenizer._convert_token_to_id_with_added_voc(tokenizer.src_lang)]]),
                ),
                axis=1
            )
            model_inputs.pop("token_type_ids")
            model_inputs["attention_mask"] = np.ones_like(model_inputs["input_ids"])

        if data_args.target_key in examples:
            targets = [normalize(ex, **normalization_kwargs) if data_args.do_normalize else ex
                        for ex in examples[data_args.target_key]]

            tokenizer_kwargs.update({"max_length": max_target_length})
            
            if is_unified and tokenizer.tgt_lang in code2script :
                targets = [UnicodeIndicTransliterator.transliterate(k, code2script[tokenizer.tgt_lang], "hi")
                            for k in targets]

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, **tokenizer_kwargs)

            if is_indicbart:
                labels["input_ids"] = np.concatenate(
                    (
                        labels["input_ids"], 
                        np.array([[eos_id, tokenizer._convert_token_to_id_with_added_voc(tokenizer.tgt_lang)]])
                    ),
                    # LID will get wrapped around through modeling_mbart
                    axis=1
                )
                
            model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=1 if is_indicbart else training_args.train_batch_size,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=1 if is_indicbart else training_args.train_batch_size,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=1 if is_indicbart else training_args.train_batch_size,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=predict_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    def extract_rouge_mid_statistics(dct):
        new_dict = {}
        for k1, v1 in dct.items():
            mid = v1.mid
            new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
        return new_dict

    def add_newline_to_end_of_each_sentence(x):
        return "\n".join(nltk.sent_tokenize(x))

    def process_decoded_lines(lines):
        if is_unified and tokenizer.tgt_lang in code2script:
            lines = [UnicodeIndicTransliterator.transliterate(k, "hi", code2script[tokenizer.tgt_lang])
                            for k in lines]

        return lines


    def calculate_rouge(
        pred_lns,
        tgt_lns,
        use_stemmer=True,
        rouge_keys=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        return_precision_and_recall=False,
        bootstrap_aggregation=True,
        newline_sep=True,
        rouge_lang=data_args.rouge_lang,
    ):
        
        logger.info("Rouge lang: " + str(rouge_lang))
        scorer = rouge_scorer.RougeScorer(
            rouge_keys, lang=rouge_lang,
            use_stemmer=use_stemmer
        )
        aggregator = scoring.BootstrapAggregator()
        for pred, tgt in zip(tgt_lns, pred_lns):
            # rougeLsum expects "\n" separated sentences within a summary
            if newline_sep:
                pred = add_newline_to_end_of_each_sentence(pred)
                tgt = add_newline_to_end_of_each_sentence(tgt)
            scores = scorer.score(pred, tgt)
            aggregator.add_scores(scores)

        if bootstrap_aggregation:
            result = aggregator.aggregate()
            if return_precision_and_recall:
                return extract_rouge_mid_statistics(result)  # here we return dict
            else:
                return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

        else:
            return aggregator._scores  # here we return defaultdict(list)

  
    def calculate_bleu(pred_lns, tgt_lns, **kwargs):
        return {
            "sacrebleu": round(
                corpus_bleu(
                    [k.strip() for k in pred_lns], 
                    [[k.strip() for k in tgt_lns]], 
                    **kwargs).score, 
                4)
        }

    metric_fn = calculate_rouge if data_args.evaluation_metric == "rouge" else calculate_bleu

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
        decoded_preds = process_decoded_lines(tokenizer.batch_decode(preds, skip_special_tokens=True))
        decoded_labels = process_decoded_lines(tokenizer.batch_decode(labels, skip_special_tokens=True))

        result = metric_fn(decoded_preds, decoded_labels)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model() 

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        results.update(metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        results.update(metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = process_decoded_lines(
                    tokenizer.batch_decode(
                        predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    all_results_path = os.path.join(training_args.output_dir, "all_results.json")
    with open(all_results_path, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()