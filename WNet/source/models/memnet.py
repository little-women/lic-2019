# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-19 11:49:16
# @Last Modified by:   liwei
# @Last Modified time: 2019-04-19 16:31:48


import torch
import torch.nn as nn
import torch.nn.functional as F

from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder
from source.modules.attention import Attention
from source.utils.criterions import NLLLoss


class MemNet(BaseModel):
    """
    symbols: vocabulary size.
    num_entities: entitiy vocabulary size.
    num_relations", 44, "relation size.
    embed_units", 300, "Size of word embedding.
    trans_units", 100, "Size of trans embedding.
    """

    def __init__(self, symbols, num_entities, num_relations, embed_units,
                 trans_units, hidden_size, padding_idx=None,  num_layers=1,
                 bidirectional=True, attn_mode='mlp', dropout=0.0, use_gpu=False):
        super(MemNet, self).__init__()

        self.symbols = symbols
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_units = embed_units
        self.trans_units = trans_units
        self.padding_idx = padding_idx

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.attn_mode = attn_mode
        self.use_gpu = use_gpu

        entity_embedder = Embedder(num_embeddings=self.num_entities,
                                   embedding_dim=self.trans_units, padding_idx=self.padding_idx)
        relation_embedder = Embedder(num_embeddings=self.num_relations,
                                     embedding_dim=self.trans_units, padding_idx=self.padding_idx)
        enc_embedder = Embedder(num_embeddings=self.symbols,
                                embedding_dim=self.embed_units, padding_idx=self.padding_idx)

        self.encoder = RNNEncoder(
            input_size=self.embed_units, hidden_size=self.hidden_size, embedder=enc_embedder, num_layers=self.num_layers,
            bidirectional=self.bidirectional, dropout=self.dropout)

        self.fact_attention = Attention(query_size=self.hidden_size,
                                        memory_size=self.hidden_size,
                                        hidden_size=self.hidden_size,
                                        mode='dot')

        self.softmax = nn.Softmax(dim=-1)

        if self.padding_idx is not None:
            self.weight = torch.ones(self.symbols)
            self.weight[self.padding_idx] = 0
        else:
            self.weight = None
        self.nll_loss = NLLLoss(weight=self.weight, ignore_index=self.padding_idx,
                                reduction='mean')

        if self.use_gpu:
            self.cuda()
            self.weight = self.weight.cuda()

    def encode(self, inputs, hidden=None):
        return outputs, dec_init_state

    def decode(self, input, state):
        return log_prob

    def forward(self, enc_inputs, dec_inputs, hidden=None):
        return outputs
