import ujson as json
import numpy as np

def get_data(filename) :
    with open(filename, 'r', encoding='utf-8') as f :
        data = json.load(f)
    return data

def load_data(opts):
    print('load data...')
    data_path = opts['data_path']

    train_data = get_data(data_path + 'train.json')
    dev_data = get_data(data_path + 'dev.json')
    word2id = get_data(data_path + 'word2id.json')
    char2id = get_data(data_path + 'char2id.json')
    pos2id = get_data(data_path + 'pos2id.json')
    ner2id = get_data(data_path + 'ner2id.json')

    opts['char_size'] = int(np.max(list(char2id.values())) + 1)
    opts['pos_size'] = int(np.max(list(pos2id.values())) + 1)
    opts['ner_size'] = int(np.max(list(ner2id.values())) + 1)

    return train_data, dev_data, word2id, char2id, opts

def get_batches(data, batch_size) :

    batches = []
    for i in range(0, len(data['context_ids']), batch_size) :
        batches.append((data['context_ids'][i:i+batch_size],
                        data['context_char_ids'][i:i+batch_size],
                        data['context_pos_ids'][i:i+batch_size],
                        data['context_ner_ids'][i:i+batch_size],
                        data['context_match_origin'][i:i+batch_size],
                        data['context_match_lower'][i:i+batch_size],
                        data['context_match_lemma'][i:i+batch_size],
                        data['context_tf'][i:i+batch_size],
                        data['ques_ids'][i:i+batch_size],
                        data['ques_char_ids'][i:i+batch_size],
                        data['ques_pos_ids'][i:i+batch_size],
                        data['ques_ner_ids'][i:i+batch_size],
                        data['y1'][i:i+batch_size],
                        data['y2'][i:i+batch_size],
                        data['id'][i:i+batch_size]))

    return batches