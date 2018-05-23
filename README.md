# FusionNet: Fusing via Fully-aware Attention with Application to Machine Comprehension
* A pytorch implementation of [FusionNet: Fusing via Fully-aware Attention with Application to Machine Comprehension](https://openreview.net/pdf?id=BJIgi_eCZ).
* this model was evaluated in the SQuAD dataset.

# Requirements
* python 3.5
* pytorch 0.3.1
* numpy
* usjon
* spacy

# data preprocess
* The preprocessing code of SQuAD dataset is based on [HKUST-KnowComp/R-Net](https://github.com/HKUST-KnowComp/R-Net).


# Quick start
1. ./download.sh
2. python3 prepro.py
3. python3 train.py

# Performance
* F1 = 83.15%
* EM = 74.47%

![scores.jpg](./scores.jpg)
