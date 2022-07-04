# BanglaNLG

This repository contains the official release of the model **"BanglaT5"** and associated downstream finetuning code and datasets introduced in the paper titled [**"BanglaNLG: Benchmarks and Resources for Evaluating Low-Resource
Natural Language Generation in Bangla"**](https://arxiv.org/abs/2205.11081).

## Table of Contents

- [BanglaNLG](#banglanlg)
  - [Table of Contents](#table-of-contents)
  - [Models](#models)
  - [Setup](#setup)
  - [Training & Evaluation](#training--evaluation)
  - [Benchmarks](#benchmarks)
  - [License](#license)
  - [Citation](#citation)

## Models

The **BanglaT5** model checkpoint is available at [Huggingface model hub](https://huggingface.co/csebuetnlp/banglat5).
  
To use this model for the supported downstream tasks in this repository see **[Training & Evaluation](#training--evaluation).**

***Note:*** This model was pretrained using a ***specific normalization pipeline*** available **[here](https://github.com/csebuetnlp/normalizer)**. All finetuning scripts in this repository uses this normalization by default. If you need to adapt the pretrained model for a different task make sure ***the text units are normalized using this pipeline before tokenizing*** to get best results. A basic example is available at the **[model page](https://huggingface.co/csebuetnlp/banglat5).**


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

|     Model          |   Params   |     MT (SacreBLEU)    |      TS (ROUGE-2)     |      QA (EM/F1)   |   BNLG score |
|--------------------|------------|-----------------------|------------------------|-------------------|--------------|
|[mT5 (base)](https://huggingface.co/google/mt5-base) | 582M  | 36.6/22.5 | 10.27 | 58.95/65.32 | 38.73 |
|[BanglaT5](https://huggingface.co/csebuetnlp/banglat5) | 247M | 38.8/25.2 | 13.66 | 68.49/74.77 | 44.18 |


The benchmarking datasets are as follows:
* **MT:** **[Machine Translation]()**
* **ATS:** **[Abstractive Text Summarization]()**
* **QA:** **[Question Answering]()**
  

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
