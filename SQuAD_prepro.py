#coding: utf-8

import random
import ujson as json
import numpy as np
import spacy

from tqdm import tqdm
from collections import Counter

nlp = spacy.load("en", parser=False)

def word_tokenize(sent):
    doc = nlp(sent)

    text = []
    tag = []
    ent = []
    for token in doc :
        text.append(token.text)
        tag.append(token.tag_)
        ent.append(token.ent_type_)

    return text, tag, ent


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def process_file(filename, data_type, word_counter, char_counter, pos_counter, ner_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens, context_tags, context_ents = word_tokenize(context)
                context_lower_tokens = [w.lower() for w in context_tokens]
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)

                context_pos_set = set(context_tags)
                context_ner_set = set(context_ents)
                for pos in context_pos_set :
                    pos_counter[pos] += 1
                for ner in context_ner_set :
                    ner_counter[ner] += 1
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens, ques_tags, ques_ents = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]

                    ques_tokens_set = set(ques_tokens)
                    match_origin = [w in ques_tokens_set for w in context_tokens]
                    ques_pos_set = set(ques_tags)
                    ques_ner_set = set(ques_ents)
                    for pos in ques_pos_set:
                        pos_counter[pos] += 1
                    for ner in ques_ner_set:
                        ner_counter[ner] += 1

                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)

                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "match_origin" : match_origin, "context_pos" : context_tags, "context_ner" : context_ents,
                               "ques_tokens": ques_tokens, "ques_pos" : ques_tags, "ques_ner" : ques_ents,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples

def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None, token2idx_dict=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.01) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(
        embedding_dict.keys(), 2)} if token2idx_dict is None else token2idx_dict
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict

def make_dict(counter) :
    NULL = "--NULL--"
    OOV = "--OOV--"
    index = 2
    token2idx_dict = {}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    for text in counter.keys() :
        if text not in token2idx_dict :
            token2idx_dict[text] = index
            index += 1

    return token2idx_dict


def build_features(examples, data_type, out_file, word2idx_dict, char2idx_dict, pos2idx_dict, ner2idx_dict, is_test=False):

    para_limit = 1000 if is_test else 400
    ques_limit = 100 if is_test else 50
    char_limit = 16

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

    print("Processing {} examples...".format(data_type))
    #writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0

    context_ids = []
    context_match_origin = []
    context_char_ids = []
    context_pos_ids = []
    context_ner_ids = []
    ques_ids = []
    ques_char_ids = []
    ques_pos_ids = []
    ques_ner_ids = []
    y1 = []
    y2 = []
    id = []
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example, is_test):
            continue

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        match_origin = np.zeros([para_limit], dtype=np.int32)
        context_pos_idxs = np.zeros([para_limit], dtype=np.int32)
        context_ner_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_pos_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_ner_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        #y1 = np.zeros([para_limit], dtype=np.float32)
        #y2 = np.zeros([para_limit], dtype=np.float32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_pos(pos) :
            if pos in pos2idx_dict :
                return pos2idx_dict[pos]
            return 1
        def _get_ner(ner) :
            if ner in ner2idx_dict :
                return ner2idx_dict[ner]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)
        for i, match in enumerate(example["match_origin"]) :
            match_origin[i] = 1 if match == True else 0

        for i, pos in enumerate(example['context_pos']) :
            context_pos_idxs[i] = _get_pos(pos)
        for i, ner in enumerate(example['context_ner']) :
            context_ner_idxs[i] = _get_pos(ner)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, pos in enumerate(example['ques_pos']) :
            ques_pos_idxs[i] = _get_pos(pos)
        for i, ner in enumerate(example['ques_ner']) :
            ques_ner_idxs[i] = _get_pos(ner)


        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = example["y1s"][-1], example["y2s"][-1]
        #y1[start], y2[end] = 1.0, 1.0

        context_ids.append(context_idxs.tolist())
        context_match_origin.append(match_origin.tolist())
        context_pos_ids.append(context_pos_idxs.tolist())
        context_ner_ids.append(context_ner_idxs.tolist())
        context_char_ids.append(context_char_idxs.tolist())
        ques_ids.append(ques_idxs.tolist())
        ques_pos_ids.append(ques_pos_idxs.tolist())
        ques_ner_ids.append(ques_ner_idxs.tolist())
        ques_char_ids.append(ques_char_idxs.tolist())
        y1.append(start)
        y2.append(end)
        id.append(example['id'])

    print("Build {} / {} instances of features in total".format(total, total_))

    data = {
        "context_ids" : context_ids,
        "context_match_origin" : context_match_origin,
        "context_char_ids" : context_char_ids,
        "context_pos_ids" : context_pos_ids,
        "context_ner_ids" : context_ner_ids,
        "ques_ids" : ques_ids,
        "ques_char_ids" : ques_char_ids,
        "ques_pos_ids" : ques_pos_ids,
        "ques_ner_ids" : ques_ner_ids,
        "y1" : y1,
        "y2" : y2,
        "id" : id,
        "total" : total
    }

    with open(out_file, 'w') as f :
        json.dump(data, f)

if __name__ == '__main__' :

    word_counter, char_counter = Counter(), Counter()
    pos_counter, ner_counter = Counter(), Counter()

    train_examples, train_eval = process_file('SQuAD/train-v1.1.json', "train", word_counter, char_counter, pos_counter, ner_counter)
    dev_examples, dev_eval = process_file('SQuAD/dev-v1.1.json', "dev", word_counter, char_counter, pos_counter, ner_counter)

    pos2id = make_dict(pos_counter)
    ner2id = make_dict(ner_counter)

    word_emb_file = 'glove/glove.840B.300d.txt'
    glove_word_size = int(2.2e6)
    glove_dim = 300
    word2id = None
    word_emb, word2id = get_embedding(word_counter, "word", emb_file = word_emb_file,
                                      size = glove_word_size, vec_size = glove_dim,
                                      token2idx_dict=word2id)


    char_emb_dim = 50
    char_size = 94
    char2id = None
    char_emb, char2id = get_embedding(char_counter, "char", size = char_size, vec_size = char_emb_dim,
                                      token2idx_dict=char2id)

    build_features(train_examples, "train",
                   'SQuAD/train.json', word2id, char2id, pos2id, ner2id)
    build_features(dev_examples, "dev",
                   'SQuAD/dev.json', word2id, char2id, pos2id, ner2id)

    with open('SQuAD/train_eval.json', 'w', encoding='utf-8') as f :
        json.dump(train_eval, f)
    with open('SQuAD/dev_eval.json', 'w', encoding='utf-8') as f :
        json.dump(dev_eval, f)

    with open('SQuAD/word_emb.json', 'w', encoding='utf-8') as f :
        json.dump(word_emb, f)

    with open('SQuAD/word2id.json', 'w', encoding='utf-8') as f :
        json.dump(word2id, f)

    with open('SQuAD/char2id.json', 'w', encoding='utf-8') as f :
        json.dump(char2id, f)

    with open('SQuAD/pos2id.json', 'w') as f :
        json.dump(pos2id, f)

    with open('SQuAD/ner2id.json', 'w') as f :
        json.dump(ner2id, f)