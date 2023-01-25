# BanglaNLG

This repository contains the official release of the model **"BanglaT5"** and associated downstream finetuning code and datasets introduced in the paper titled [**"BanglaNLG and BanglaT5: Benchmarks and Resources for Evaluating Low-Resource
Natural Language Generation in Bangla"**](https://arxiv.org/abs/2205.11081) accepted in the 17th Conference of the European Chapter
of the Association for Computational Linguistics (EACL 2023).

## Table of Contents

- [BanglaNLG](#banglanlg)
  - [Table of Contents](#table-of-contents)
  - [Models](#models)
  - [Datasets](#datasets)
  - [Setup](#setup)
  - [Training & Evaluation](#training--evaluation)
  - [Benchmarks](#benchmarks)
  - [License](#license)
  - [Citation](#citation)

## Models

The **BanglaT5** model checkpoint is available at [Huggingface model hub](https://huggingface.co/csebuetnlp/banglat5).
  
To use this model for the supported downstream tasks in this repository see **[Training & Evaluation](#training--evaluation).**

We also release the following finetuned checkpoints:
Model Name        |Task name|
--------------|-------------|
[banglat5_nmt_bn_en](https://huggingface.co/csebuetnlp/banglat5_nmt_bn_en)| Bengali-English MT |
[banglat5_nmt_en_bn](https://huggingface.co/csebuetnlp/banglat5_nmt_en_bn)| English-Bengali MT |



***Note:*** This model was pretrained using a ***specific normalization pipeline*** available **[here](https://github.com/csebuetnlp/normalizer)**. All finetuning scripts in this repository uses this normalization by default. If you need to adapt the pretrained model for a different task make sure ***the text units are normalized using this pipeline before tokenizing*** to get best results. A basic example is available at the **[model page](https://huggingface.co/csebuetnlp/banglat5).**

## Datasets

The benchmarking datasets are as follows:
* **MT:** **[Machine Translation](https://github.com/csebuetnlp/banglanmt#datasets)**
* **TS:** **[Abstractive Text Summarization](https://huggingface.co/datasets/csebuetnlp/xlsum)**
* **QA:** **[Question Answering](https://huggingface.co/datasets/csebuetnlp/squad_bn)**
* **MTD:** **[Multi Turn Dialogue Generation](https://drive.google.com/file/d/1qPmNN6qA4evbh4cD_BDDTCFOwMu4H2JS/view?usp=sharing)** (**Introduced in this work**)
* **NHG:** **[News Headline Generation](https://huggingface.co/datasets/csebuetnlp/xlsum)**
* **XLS:** **[Cross-lingual Summarization](https://huggingface.co/datasets/csebuetnlp/CrossSum)**

Please see the **[BanglaBERT repository](https://github.com/csebuetnlp/banglabert#datasets)** to access the pretraining corpus.

## Setup

For installing the necessary requirements, use the following snippet
```bash
$ git clone https://github.com/csebuetnlp/BanglaNLG
$ cd BanglaNLG/
$ conda create python==3.7.9 pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch -p ./env
$ conda activate ./env # or source activate ./env (for older versions of anaconda)
$ bash setup.sh 
```
* Use the newly created environment for running the scripts in this repository.

## Training & Evaluation

While all tasks we consider are modeled as seq2seq tasks, some tasks need specific data preprocessing for preparing the input and output sequences.
See below for task-specific finetuning/inference scripts:

* **[Sequence To Sequence](seq2seq/).**
  - For general sequence to sequence tasks such as
    - Machine Translation
    - Text Summarization 
    - News Headline Generation etc.
- **[Question Answering](question_answering/).**
    - For tasks such as,
      - Extractive Question Answering
      - Open-domain Question Answering
- **[Dialogue Generation](dialogue_generation/).**
    - For tasks such as,
      - Single Turn Dialogue
      - Multi Turn Dialogue
  
## Benchmarks
 
* Supervised fine-tuning

|     Model          |   Params   |     MT (SacreBLEU)    |      TS (ROUGE-2)     |      QA (EM/F1)   |   MTD (SacreBLEU-1)  |  NHG (ROUGE-2) |  XLS (ROUGE-2) |
|--------------------|------------|-----------------------|------------------------|-------------------|--------------------|----------------|----------------|
|[mT5 (base)](https://huggingface.co/google/mt5-base) | 582M  | 30.1/17.2 | 10.3 | 59.0/65.3 | 17.5 |  9.6 | 2.7/0.7 |
|[XLM-ProphetNet](https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased) | 616M | 27.5/15.4 | 7.8 | 53.0/57.3 | 20.0 | 9.5 | 6.2/2.7 |
|[mBART-50](https://huggingface.co/facebook/mbart-large-50) | 611M | 29.7/15.5 | 10.4 | 53.4/58.9 | 18.5 | 11.2 | 5.4/3.7 |
|[IndicBART (unified)](https://huggingface.co/ai4bharat/IndicBART) | 244M | 28.1/16.6 | 8.9 | 59.6/65.6 | 14.8 | 7.9 | 6.3/2.5 |
|[IndicBART (separate)](https://huggingface.co/ai4bharat/IndicBARTSS) | 244M | 27.5/15.7 | 9.2 | 55.3/61.2 | 14.1 | 9.1 | 5.3/2.4 |
|[BanglaT5](https://huggingface.co/csebuetnlp/banglat5) | 247M | 31.3/17.4 | 13.7 | 68.5/74.8 | 19.0 | 13.8 | 6.4/4.0 |
  
## License
Contents of this repository are restricted to non-commercial research purposes only under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

## Citation
If you use any of the datasets, models or code modules, please cite the following paper:
```
@article{bhattacharjee2022banglanlg,
  author    = {Abhik Bhattacharjee and Tahmid Hasan and Wasi Uddin Ahmad and Rifat Shahriyar},
  title     = {BanglaNLG: Benchmarks and Resources for Evaluating Low-Resource Natural Language Generation in Bangla},
  journal   = {CoRR},
  volume    = {abs/2205.11081},
  year      = {2022},
  url       = {https://arxiv.org/abs/2205.11081},
  eprinttype = {arXiv},
  eprint    = {2205.11081}
}
```
