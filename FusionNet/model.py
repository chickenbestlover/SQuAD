import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import ujson as json

from torch.autograd import Variable
from utils.layers import StackedLSTM, Dropout, FullAttention, WordAttention, Summ, PointerNet
from utils.dataset import get_data
from utils.cove import MTLSTM
from utils.evaluate import evaluate

class FusionNet(nn.Module):
    def __init__(self, opts):
        super(FusionNet, self).__init__()
        self.opts = opts
        self.build_model()

    def build_model(self):
        opts = self.opts
        print('load embedding...')
        word_emb = np.array(get_data(opts['data_path'] + 'word_emb.json'), dtype=np.float32)
        word_dim = word_emb.shape[1]
        self.word_embeddings = nn.Embedding(word_emb.shape[0], word_dim, padding_idx=0)
        self.word_embeddings.weight.data = torch.from_numpy(word_emb)

        #self.char_embeddings = nn.Embedding(opts['char_size'], opts['char_dim'], padding_idx=0)
        self.pos_embeddings = nn.Embedding(opts['pos_size'], opts['pos_dim'], padding_idx=0)
        self.ner_embeddings = nn.Embedding(opts['ner_size'], opts['ner_dim'], padding_idx=0)

        #char_hidden_size = opts['char_hidden_size']
        cove_dim = 600
        pos_dim = opts['pos_dim']
        ner_dim = opts['ner_dim']
        hidden_size = opts['hidden_size']
        dropout = opts['dropout']
        attention_size = opts['attention_size']

        """self.char_rnn = StackedLSTM(input_size=opts['char_dim'],
                                    hidden_size=char_hidden_size,
                                    num_layers=1,
                                    dropout=dropout)"""

        opt = {
            'vocab_size': word_emb.shape[0],
            'embedding_dim': word_dim,
            'MTLSTM_path': 'utils/MT-LSTM.pth'
        }
        self.cove_rnn = MTLSTM(opt,
                               embedding=torch.from_numpy(word_emb))

        low_p_word_size = word_dim + word_dim + cove_dim + opts['pos_dim'] + opts['ner_dim'] + 1
        low_q_word_size = word_dim + cove_dim + opts['pos_dim'] + opts['ner_dim']

        self.word_attention_layer = WordAttention(input_size = word_dim,
                                                  hidden_size = attention_size,
                                                  dropout = dropout)

        self.low_passage_rnn = StackedLSTM(input_size=low_p_word_size,
                                                hidden_size=hidden_size,
                                                num_layers=1,
                                                dropout=dropout)

        self.low_ques_rnn = StackedLSTM(input_size=low_q_word_size,
                                             hidden_size=hidden_size,
                                             num_layers=1,
                                             dropout=dropout)

        high_p_word_size = 2 * hidden_size
        high_q_word_size = 2 * hidden_size

        self.high_passage_rnn = StackedLSTM(input_size=high_p_word_size,
                                                hidden_size=hidden_size,
                                                num_layers=1,
                                                dropout=dropout)

        self.high_ques_rnn = StackedLSTM(input_size=high_q_word_size,
                                                hidden_size=hidden_size,
                                                num_layers=1,
                                                dropout=dropout)

        und_q_word_size = 2 * (2 * hidden_size)

        self.und_ques_rnn = StackedLSTM(input_size=und_q_word_size,
                                                hidden_size=hidden_size,
                                                num_layers=1,
                                                dropout=dropout)

        attention_inp_size = word_dim + cove_dim + 2 * (2 * hidden_size)

        self.low_attention_layer = FullAttention(input_size = attention_inp_size,
                                                 hidden_size = attention_size,
                                                 dropout = dropout)

        self.high_attention_layer = FullAttention(input_size=attention_inp_size,
                                                  hidden_size=attention_size,
                                                  dropout=dropout)

        self.und_attention_layer = FullAttention(input_size=attention_inp_size,
                                                  hidden_size=attention_size,
                                                  dropout=dropout)

        fuse_inp_size = 5 * (2 * hidden_size)

        self.fuse_rnn = StackedLSTM(input_size = fuse_inp_size,
                                    hidden_size = hidden_size,
                                    num_layers = 1,
                                    dropout = dropout)

        self_attention_inp_size = word_dim + cove_dim + pos_dim + ner_dim + 6 * (2 * hidden_size)

        self.self_attention_layer = FullAttention(input_size=self_attention_inp_size,
                                                  hidden_size=attention_size,
                                                  dropout=dropout)

        self.self_rnn = StackedLSTM(input_size = 2 * (2 * hidden_size),
                                    hidden_size = hidden_size,
                                    num_layers = 1,
                                    dropout = dropout)

        self.summ_layer = Summ(input_size=2 * hidden_size,
                               dropout=dropout)

        self.pointer_layer = PointerNet(input_size=2 * hidden_size,
                                        dropout=dropout)

    def compute_mask(self, x):
        mask = torch.eq(x, 0)
        if self.opts['cuda']:
            mask = mask.cuda()
        return mask

    def prepare_data(self, batch_data):
        """
        batch_data[0] : passage_ids,
        batch_data[1] : passage_char_ids,
        batch_data[2] : passage_pos_ids,
        batch_data[3] : passage_ner_ids,
        batch_data[4] : passage_match_origin,
        batch_data[5] : ques_ids,
        batch_data[6] : ques_char_ids,
        batch_data[7] : ques_pos_ids,
        batch_data[8] : ques_ner_ids,
        batch_data[9] : y1,
        batch_data[10] : y2,
        batch_data[11] : id
        """

        passage_ids = Variable(torch.LongTensor(batch_data[0]))
        passage_char_ids = Variable(torch.LongTensor(batch_data[1]))
        passage_pos_ids = Variable(torch.LongTensor(batch_data[2]))
        passage_ner_ids = Variable(torch.LongTensor(batch_data[3]))
        passage_match = Variable(torch.FloatTensor(batch_data[4]))

        ques_ids = Variable(torch.LongTensor(batch_data[5]))
        ques_char_ids = Variable(torch.LongTensor(batch_data[6]))
        ques_pos_ids = Variable(torch.LongTensor(batch_data[7]))
        ques_ner_ids = Variable(torch.LongTensor(batch_data[8]))

        y1 = Variable(torch.LongTensor(batch_data[9]))
        y2 = Variable(torch.LongTensor(batch_data[10]))

        if self.opts['cuda']:
            passage_ids = passage_ids.cuda()
            passage_char_ids = passage_char_ids.cuda()
            passage_pos_ids = passage_pos_ids.cuda()
            passage_ner_ids = passage_ner_ids.cuda()
            passage_match = passage_match.cuda()
            ques_ids = ques_ids.cuda()
            ques_char_ids = ques_char_ids.cuda()
            ques_pos_ids = ques_pos_ids.cuda()
            ques_ner_ids = ques_ner_ids.cuda()
            y1 = y1.cuda()
            y2 = y2.cuda()

        p_mask = self.compute_mask(passage_ids)
        q_mask = self.compute_mask(ques_ids)

        passage_maxlen = int(torch.max(p_mask.eq(0).long().sum(1), 0)[0])
        ques_maxlen = int(torch.max(q_mask.eq(0).long().sum(1), 0)[0])

        passage_ids = passage_ids[:, :passage_maxlen]
        passage_char_ids = passage_char_ids[:, :passage_maxlen]
        passage_pos_ids = passage_pos_ids[:, :passage_maxlen]
        passage_ner_ids = passage_ner_ids[:, :passage_maxlen]
        passage_match = passage_match[:, :passage_maxlen]
        p_mask = p_mask[:, :passage_maxlen]
        ques_ids = ques_ids[:, :ques_maxlen]
        ques_char_ids = ques_char_ids[:, :ques_maxlen]
        ques_pos_ids = ques_pos_ids[:, :ques_maxlen]
        ques_ner_ids = ques_ner_ids[:, :ques_maxlen]
        q_mask = q_mask[:, :ques_maxlen]

        batch_data = {
            "passage_ids": passage_ids,
            "passage_char_ids": passage_char_ids,
            "passage_pos_ids": passage_pos_ids,
            "passage_ner_ids": passage_ner_ids,
            "passage_match": passage_match,
            "p_mask": p_mask,
            "ques_ids": ques_ids,
            "ques_char_ids": ques_char_ids,
            "ques_pos_ids": ques_pos_ids,
            "ques_ner_ids": ques_ner_ids,
            "q_mask": q_mask,
            "y1": y1,
            "y2": y2,
            "id" : batch_data[11],
        }

        return batch_data

    def encoding_forward(self, batch_data):

        passage_ids = batch_data['passage_ids']
        #passage_char_ids = batch_data['passage_char_ids']
        passage_pos_ids = batch_data['passage_pos_ids']
        passage_ner_ids = batch_data['passage_ner_ids']
        passage_match = batch_data['passage_match'].unsqueeze(2).float()
        p_mask = batch_data['p_mask']
        p_pad_mask = p_mask.eq(0).unsqueeze(2).float()

        ques_ids = batch_data['ques_ids']
        #ques_char_ids = batch_data['ques_char_ids']
        ques_pos_ids = batch_data['ques_pos_ids']
        ques_ner_ids = batch_data['ques_ner_ids']
        q_mask = batch_data['q_mask']
        q_pad_mask = q_mask.eq(0).unsqueeze(2).float()

        opts = self.opts
        dropout = opts['dropout']
        hidden_size = opts['hidden_size']
        char_hidden_size = opts['char_hidden_size']

        """"### character ###

        passage_char_emb = self.char_embeddings(passage_char_ids.view(-1, passage_char_ids.size(2)))
        ques_char_emb = self.char_embeddings(ques_char_ids.view(-1, ques_char_ids.size(2)))

        passage_char_states = self.char_rnn(passage_char_emb, self.training)[:, -1].contiguous().view(-1, passage_ids.size(1), 2 * char_hidden_size) * p_pad_mask
        ques_char_states = self.char_rnn(ques_char_emb, self.training)[:, -1].contiguous().view(-1, ques_ids.size(1), 2 * char_hidden_size) * q_pad_mask"""

        ### cove ###
        _, passage_cove = self.cove_rnn(passage_ids, p_mask)
        _, ques_cove = self.cove_rnn(ques_ids, q_mask)

        ### embedding ###
        passage_emb = self.word_embeddings(passage_ids)
        passage_pos_emb = self.pos_embeddings(passage_pos_ids)
        passage_ner_emb = self.ner_embeddings(passage_ner_ids)

        ques_emb = self.word_embeddings(ques_ids)
        ques_pos_emb = self.pos_embeddings(ques_pos_ids)
        ques_ner_emb = self.ner_embeddings(ques_ner_ids)

        ### embedding dropout ###
        passage_emb = Dropout(passage_emb, dropout, self.training)
        ques_emb = Dropout(ques_emb, dropout, self.training)
        passage_cove = Dropout(passage_cove, dropout, self.training)
        ques_cove = Dropout(ques_cove, dropout, self.training)

        ### Word Attention ###
        word_attention_outputs = self.word_attention_layer(passage_emb, p_mask, ques_emb, q_mask, self.training) * p_pad_mask

        p_word_inp = torch.cat([passage_emb, passage_cove, passage_pos_emb, passage_ner_emb, word_attention_outputs, passage_match], dim=2)
        q_word_inp = torch.cat([ques_emb, ques_cove, ques_pos_emb, ques_ner_emb], dim=2)

        low_passage_states = self.low_passage_rnn(p_word_inp, self.training) * p_pad_mask
        low_ques_states = self.low_ques_rnn(q_word_inp, self.training) * q_pad_mask

        high_passage_states = self.high_passage_rnn(low_passage_states, self.training) * p_pad_mask
        high_ques_states = self.high_ques_rnn(low_ques_states, self.training) * q_pad_mask

        und_ques_inp = torch.cat([low_ques_states, high_ques_states], dim=2)
        und_ques_states = self.und_ques_rnn(und_ques_inp, self.training) * q_pad_mask


        ### Full Attention ###

        passage_HoW = torch.cat([passage_emb, passage_cove, low_passage_states, high_passage_states], dim=2)
        ques_HoW = torch.cat([ques_emb, ques_cove, low_ques_states, high_ques_states], dim=2)

        low_attention_outputs = self.low_attention_layer(passage_HoW, p_mask, ques_HoW, q_mask, low_ques_states, self.training) * p_pad_mask
        high_attention_outputs = self.high_attention_layer(passage_HoW, p_mask, ques_HoW, q_mask, high_ques_states, self.training) * p_pad_mask
        und_attention_outputs = self.und_attention_layer(passage_HoW, p_mask, ques_HoW, q_mask, und_ques_states, self.training) * p_pad_mask

        fuse_inp = torch.cat([low_passage_states, high_passage_states, low_attention_outputs, high_attention_outputs, und_attention_outputs], dim = 2)

        fused_passage_states = self.fuse_rnn(fuse_inp, self.training) * p_pad_mask

        ### Self Full Attention ###

        passage_HoW = torch.cat([passage_emb, passage_cove, passage_pos_emb, passage_ner_emb, low_passage_states, high_passage_states, low_attention_outputs, high_attention_outputs, und_attention_outputs, fused_passage_states], dim=2)

        self_attention_outputs = self.self_attention_layer(passage_HoW, p_mask, passage_HoW, p_mask, fused_passage_states, self.training) * p_pad_mask

        self_inp = torch.cat([fused_passage_states, self_attention_outputs], dim=2)

        und_passage_states = self.self_rnn(self_inp, self.training) * p_pad_mask

        return und_passage_states, p_mask, und_ques_states, q_mask


    def decoding_forward(self, und_passage_states, p_mask, und_ques_states, q_mask) :
        ### Pointer Network ###

        beta = self.summ_layer(und_ques_states, q_mask, self.training)
        init_states = torch.bmm(beta.unsqueeze(1), und_ques_states).squeeze(1)

        logits1, logits2 = self.pointer_layer.forward(und_passage_states, p_mask, init_states, self.training)
        return logits1, logits2

    def compute_loss(self, logits1, logits2, y1, y2) :

        loss1 = F.cross_entropy(logits1, y1)
        loss2 = F.cross_entropy(logits2, y2)
        loss = loss1 + loss2
        return loss

    def forward(self, batch_data):
        batch_data = self.prepare_data(batch_data)
        und_passage_states, p_mask, und_ques_states, q_mask = self.encoding_forward(batch_data)
        logits1, logits2 = self.decoding_forward(und_passage_states, p_mask, und_ques_states, q_mask)
        loss = self.compute_loss(logits1, logits2, batch_data['y1'], batch_data['y2'])
        del und_passage_states, p_mask, und_ques_states, q_mask, logits1, logits2
        return loss


    def get_predictions(self, logits1, logits2, maxlen=15) :
        batch_size, P = logits1.size()
        outer = torch.matmul(F.softmax(logits1, -1).unsqueeze(2),
                             F.softmax(logits2, -1).unsqueeze(1))

        band_mask = Variable(torch.zeros(P, P))

        if self.opts['cuda'] :
            band_mask = band_mask.cuda()

        for i in range(P) :
            band_mask[i, i:max(i+maxlen, P)].data.fill_(1.0)

        band_mask = band_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        outer = outer * band_mask

        yp1 = torch.max(torch.max(outer, 2)[0], 1)[1]
        yp2 = torch.max(torch.max(outer, 1)[0], 1)[1]

        return yp1, yp2


    def convert_tokens(self, eval_file, qa_id, pp1, pp2) :
        answer_dict = {}
        remapped_dict = {}
        for qid, p1, p2 in zip(qa_id, pp1, pp2) :

            p1 = int(p1)
            p2 = int(p2)
            context = eval_file[str(qid)]["context"]
            spans = eval_file[str(qid)]["spans"]
            uuid = eval_file[str(qid)]["uuid"]
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            answer_dict[str(qid)] = context[start_idx : end_idx]
            remapped_dict[uuid] = context[start_idx : end_idx]
        return answer_dict, remapped_dict

    def Evaluate(self, batches, eval_file=None, answer_file = None) :
        print ('Start evaluate...')

        with open(eval_file, 'r') as f :
            eval_file = json.load(f)

        answer_dict = {}
        remapped_dict = {}

        for batch in batches :
            batch_data = self.prepare_data(batch)
            und_passage_states, p_mask, und_ques_states, q_mask = self.encoding_forward(batch_data)
            logits1, logits2 = self.decoding_forward(und_passage_states, p_mask, und_ques_states, q_mask)
            y1, y2 = self.get_predictions(logits1, logits2)
            qa_id = batch_data['id']
            answer_dict_, remapped_dict_ = self.convert_tokens(eval_file, qa_id, y1, y2)
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)

        metrics = evaluate(eval_file, answer_dict)
        with open(answer_file, 'w') as f:
            json.dump(remapped_dict, f)
        print("Exact Match: {}, F1: {}".format(
            metrics['exact_match'], metrics['f1']))

        return metrics['exact_match'], metrics['f1']