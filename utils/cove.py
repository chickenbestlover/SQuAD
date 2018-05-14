import os

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class MTLSTM(nn.Module):
    def __init__(self, opt, embedding=None, padding_idx=0):
        """Initialize an MTLSTM

        Arguments:
            embedding (Float Tensor): If not None, initialize embedding matrix with specified embedding vectors
        """
        super(MTLSTM, self).__init__()

        self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_dim'], padding_idx=padding_idx)
        if embedding is not None:
            self.embedding.weight.data = embedding

        state_dict = torch.load(opt['MTLSTM_path'])
        self.rnn1 = nn.LSTM(300, 300, num_layers=1, bidirectional=True)
        self.rnn2 = nn.LSTM(600, 300, num_layers=1, bidirectional=True)

        state_dict1 = dict([(name, param.data) if isinstance(param, nn.Parameter) else (name, param)
                        for name, param in state_dict.items() if '0' in name])
        state_dict2 = dict([(name.replace('1', '0'), param.data) if isinstance(param, nn.Parameter) else (name.replace('1', '0'), param)
                        for name, param in state_dict.items() if '1' in name])
        self.rnn1.load_state_dict(state_dict1)
        self.rnn2.load_state_dict(state_dict2)

        for p in self.embedding.parameters():
            p.requires_grad = False
        for p in self.rnn1.parameters():
            p.requires_grad = False
        for p in self.rnn2.parameters():
            p.requires_grad = False

        self.output_size = 600

    def setup_eval_embed(self, eval_embed, padding_idx=0):
        """Allow evaluation vocabulary size to be greater than training vocabulary size

        Arguments:
            eval_embed (Float Tensor): Initialize eval_embed to be the specified embedding vectors
        """
        self.eval_embed = nn.Embedding(eval_embed.size(0), eval_embed.size(1), padding_idx = padding_idx)
        self.eval_embed.weight.data = eval_embed

        for p in self.eval_embed.parameters():
            p.requires_grad = False

    def forward(self, x_idx, x_mask):
        """A pretrained MT-LSTM (McCann et. al. 2017).
        This LSTM was trained with 300d 840B GloVe on the WMT 2017 machine translation dataset.

        Arguments:
            x_idx (Long Tensor): a Long Tensor of size (batch * len).
            x_mask (Byte Tensor): a Byte Tensor of mask for the input tensor (batch * len).
        """
        emb = self.embedding #if self.training else self.eval_embed
        x_hiddens = emb(x_idx)

        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        lens, indices = torch.sort(lengths, 0, True)
        output1, _ = self.rnn1(pack(x_hiddens[indices], lens.tolist(), batch_first=True))
        output2, _ = self.rnn2(output1)

        output1 = unpack(output1, batch_first=True)[0]
        output2 = unpack(output2, batch_first=True)[0]

        _, _indices = torch.sort(indices, 0)
        output1 = output1[_indices]
        output2 = output2[_indices]

        return output1, output2