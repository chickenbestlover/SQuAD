import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class StackedLSTM(nn.Module) :

    def __init__(self, input_size, hidden_size, num_layers, dropout, concat = True, use_cuda = True) :
        super(StackedLSTM, self).__init__()
        self.use_cuda = use_cuda
        self.dropout = dropout
        self.concat = concat
        self.num_layers = num_layers
        self.rnns = nn.ModuleList()

        for layer in range(num_layers) :
            self.rnns.append(nn.LSTM(input_size = input_size if layer == 0 else 2 * hidden_size,
                                     hidden_size = hidden_size,
                                     num_layers = 1,
                                     dropout = 0,
                                     batch_first=True,
                                     bidirectional=True))

    def forward(self, x, is_training) :

        outputs = [x]

        for layer in range(self.num_layers) :
            inp = outputs[layer]
            d_inp = Dropout(inp, self.dropout, is_training, use_cuda = self.use_cuda)

            out, _ = self.rnns[layer](d_inp)
            outputs.append(out)

        outputs = outputs[1:]
        if self.concat :
            return torch.cat(outputs, dim=2)
        else :
            return outputs[-1]

class FullAttention(nn.Module) :

    def __init__(self, input_size, hidden_size, dropout, use_cuda = True):
        super(FullAttention, self).__init__()
        self.use_cuda = use_cuda
        self.dropout = dropout
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.U = nn.Linear(input_size, hidden_size, bias=False)
        self.D = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)

        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform(self.U.weight.data)

    def forward(self, passage, p_mask, question, q_mask, rep, is_training):

        if is_training :
            keep_prob = 1.0 - self.dropout
            drop_mask = Dropout(passage, self.dropout, is_training, return_mask=True, use_cuda = self.use_cuda)
            d_passage = torch.div(passage, keep_prob) * drop_mask
            d_ques = torch.div(question, keep_prob) * drop_mask
        else :
            d_passage = passage
            d_ques = question

        Up = F.relu(self.U(d_passage))
        Uq = F.relu(self.U(d_ques))
        D = self.D.expand_as(Uq)

        Uq = D * Uq

        scores = Up.bmm(Uq.transpose(2, 1))

        mask = q_mask.unsqueeze(1).repeat(1, passage.size(1), 1)
        scores.data.masked_fill_(mask.data, -float('inf'))
        alpha = F.softmax(scores, 2)
        output = torch.bmm(alpha, rep)

        return output

class WordAttention(nn.Module) :

    def __init__(self, input_size, hidden_size, dropout, use_cuda = True) :
        super(WordAttention, self).__init__()
        self.use_cuda = use_cuda
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size, hidden_size)
        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform(self.W.weight.data)
        self.W.bias.data.fill_(0.1)

    def forward(self, passage, p_mask, question, q_mask, is_training):

        if is_training :
            keep_prob = 1.0 - self.dropout
            drop_mask = Dropout(passage, self.dropout, is_training, return_mask = True, use_cuda = self.use_cuda)
            d_passage = torch.div(passage, keep_prob) * drop_mask
            d_ques = torch.div(question, keep_prob) * drop_mask
        else :
            d_passage = passage
            d_ques = question

        Wp = F.relu(self.W(d_passage))
        Wq = F.relu(self.W(d_ques))

        scores = torch.bmm(Wp, Wq.transpose(2, 1))

        mask = q_mask.unsqueeze(1).repeat(1, passage.size(1), 1)
        scores.data.masked_fill_(mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=2)
        output = torch.bmm(alpha, question)

        return output

class Summ(nn.Module) :

    def __init__(self, input_size, dropout, use_cuda = True) :
        super(Summ, self).__init__()
        self.use_cuda = use_cuda
        self.dropout = dropout
        self.w = nn.Linear(input_size, 1)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.w.weight.data)
        self.w.bias.data.fill_(0.1)

    def forward(self, x, mask, is_training) :

        d_x = Dropout(x, self.dropout, is_training, use_cuda = self.use_cuda)
        beta = self.w(d_x).squeeze(2)
        beta.data.masked_fill_(mask.data, -float('inf'))
        beta = F.softmax(beta, 1)
        output = torch.bmm(beta.unsqueeze(1), x).squeeze(1)
        return output

class PointerNet(nn.Module) :

    def __init__(self, input_size, dropout, use_cuda = True) :
        super(PointerNet, self).__init__()
        self.use_cuda = use_cuda
        self.dropout = dropout

        self.W_s = nn.Linear(input_size, input_size)
        self.W_e = nn.Linear(input_size, input_size)
        self.rnn = nn.GRUCell(input_size, input_size)

        self.init_weights()

    def init_weights(self) :

        nn.init.xavier_uniform(self.W_s.weight.data)
        self.W_s.bias.data.fill_(0.1)
        nn.init.xavier_uniform(self.W_e.weight.data)
        self.W_e.bias.data.fill_(0.1)

    def forward(self, self_states, p_mask, init_states, is_training) :

        d_init_states = Dropout(init_states.unsqueeze(1), self.dropout, is_training, use_cuda = self.use_cuda).squeeze(1)
        P0 = self.W_s(d_init_states)
        logits1 = torch.bmm(P0.unsqueeze(1), self_states.transpose(2, 1)).squeeze(1)
        logits1.data.masked_fill_(p_mask.data, -float('inf'))
        P_s = F.softmax(logits1, 1)

        rnn_input = torch.bmm(P_s.unsqueeze(1), self_states).squeeze(1)
        d_rnn_input = Dropout(rnn_input.unsqueeze(1), self.dropout, is_training, use_cuda = self.use_cuda).squeeze(1)
        end_states = self.rnn(d_rnn_input, init_states)
        d_end_states = Dropout(end_states.unsqueeze(1), self.dropout, is_training, use_cuda = self.use_cuda).squeeze(1)

        P1 = self.W_e(d_end_states)
        logits2 = torch.bmm(P1.unsqueeze(1), self_states.transpose(2, 1)).squeeze(1)
        logits2.data.masked_fill_(p_mask.data, -float('inf'))
        return logits1, logits2

def Dropout(x, dropout, is_train, return_mask = False, var=True, use_cuda=True) :

    if not var :
        return F.dropout(x, dropout, is_train)

    if dropout > 0.0 and is_train :
        shape = x.size()
        keep_prob = 1.0 - dropout
        random_tensor = keep_prob
        tmp = Variable(torch.FloatTensor(shape[0], 1, shape[2]))
        if use_cuda :
            tmp = tmp.cuda()
        nn.init.uniform(tmp)
        random_tensor += tmp
        binary_tensor = torch.floor(random_tensor)
        x = torch.div(x, keep_prob) * binary_tensor

    if return_mask :
        return binary_tensor


    return x