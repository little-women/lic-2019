# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-29 20:43:20
# @Last Modified by:   liwei
# @Last Modified time: 2019-04-29 23:02:17

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder
from source.modules.decoders.rnn_decoder import RNNDecoder
from source.utils.criterions import NLLLoss
from source.utils.metrics import accuracy
from source.utils.misc import Pack


class PointerNet(BaseModel):
    """docstring for PointerNet"""

    def __init__(self, vocab_size, embed_units, hidden_size,
                 padding_idx=None, num_layers=1, bidirectional=True,
                 attn_mode='mlp', dropout=0.0,
                 use_gpu=False):
        super(PointerNet, self).__init__()

        self.vocab_size = vocab_size
        self.embed_units = embed_units
        self.padding_idx = padding_idx

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.attn_mode = attn_mode
        self.use_gpu = use_gpu

        embedder = Embedder(num_embeddings=self.vocab_size,
                            embedding_dim=self.embed_units,
                            padding_idx=self.padding_idx)

        self.fact_encoder = RNNEncoder(input_size=self.embed_units,
                                       hidden_size=self.hidden_size,
                                       embedder=embedder,
                                       num_layers=self.num_layers,
                                       bidirectional=self.bidirectional,
                                       dropout=self.dropout)
        self.hist_encoder = RNNEncoder(input_size=self.embed_units,
                                       hidden_size=self.hidden_size,
                                       embedder=embedder,
                                       num_layers=self.num_layers,
                                       bidirectional=self.bidirectional,
                                       dropout=self.dropout)

        self.decoder = RNNDecoder(input_size=self.embed_units,
                                  hidden_size=self.hidden_size,
                                  output_size=self.vocab_size,
                                  embedder=embedder,
                                  attn_mode=self.attn_mode,
                                  attn_hidden_size=self.hidden_size,
                                  memory_size=self.hidden_size,
                                  feature_size=None,
                                  dropout=self.dropout)

        if self.padding_idx is not None:
            self.weight = torch.ones(self.vocab_size)
            self.weight[self.padding_idx] = 0
        else:
            self.weight = None
        self.nll_loss = NLLLoss(weight=self.weight,
                                ignore_index=self.padding_idx,
                                reduction='mean')

        if self.use_gpu:
            self.cuda()
            self.weight = self.weight.cuda()

    def encode(self, inputs, hidden=None):
        outputs = Pack()
        hist_inputs = _, lengths = inputs.src[0][:, 1:-1], inputs.src[1] - 2
        # (batch_size, seq_length, hidden_size*num_directions)
        # (num_layers, batch_size, num_directions * hidden_size)
        hist_outputs, hist_hidden = self.hist_encoder(hist_inputs, hidden)

        # knowledge
        batch_size, sent_num, sent = inputs.cue[0].size()
        tmp_len = inputs.cue[1]
        tmp_len[tmp_len > 0] -= 2
        fact_inputs = inputs.cue[0].view(-1, sent)[:, 1:-1], tmp_len.view(-1)
        fact_enc_outputs, fact_enc_hidden = self.fact_encoder(
            fact_inputs, hidden)
        # print(fact_enc_outputs.size())

        fact_outputs = fact_enc_outputs.view(
            batch_size, sent_num * (sent - 2), -1)

        # # (batch_size, sent_num, hidden_size)
        # fact_hidden = fact_enc_hidden[-1].view(batch_size, sent_num, -1)
        # # (batch_size, hidden_size)
        # fact_hidden = torch.sum(fact_hidden, 1).squeeze(1)

        # print(hist_outputs.size(), fact_outputs.size())

        dec_init_state = self.decoder.initialize_state(
            hist_hidden=hist_hidden[-1],
            # fact_hidden=fact_hidden,
            attn_fact=fact_outputs if self.attn_mode else None,
            attn_hist=hist_outputs if self.attn_mode else None,
            fact_lengths=lengths if self.attn_mode else None,
            hist_lengths=tmp_len if self.attn_mode else None)

        return outputs, dec_init_state
