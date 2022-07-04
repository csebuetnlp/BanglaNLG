#!/bin/bash

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/huggingface/transformers.git
cd transformers/
git checkout 7a26307e3186926373cf9129248c209ab869148b
pip install --upgrade ./
cd ../

pip install --upgrade -r requirements.txt