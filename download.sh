#!/usr/bin/env bash

# Download SQuAD dataset
mkdir -p SQuAD
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O SQuAD/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O SQuAD/dev-v1.1.json

# Download GloVe
mkdir -p glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove/glove.840B.300d.zip
unzip $GLOVE_DIR/glove.840B.300d.zip -d glove

# Download Spacy language models
python3 -m spacy download en