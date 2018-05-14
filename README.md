# FusionNet: Fusing via Fully-aware Attention with Application to Machine Comprehension
* 위 코드는 해당 논문( https://openreview.net/forum?id=BJIgi_eCZ )을 구현한 코드 입니다.

# Requirements
* python 3.5
* pytorch 0.3.1
* numpy
* usjon
* spacy

# data preprocess
SQuAD data의 preprocess는 https://github.com/HKUST-KnowComp/R-Net 의 prepro.py를 수정해서 사용하였습니다.


# Quick start
1. ./download.sh
2. python3 SQuAD_prepro.py
3. python3 train.py

# 성능표
* F1 = 83.15%
* EM = 74.47%

![scores.jpg](./scores.jpg)
