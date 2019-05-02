# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-30 18:30:33
# @Last Modified by:   liwei
# @Last Modified time: 2019-05-02 17:46:14


from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.utils.criterions import NLLLoss
from source.utils.metrics import accuracy
from source.utils.misc import Pack


class DAWNet(BaseModel):
    """
    docstring for DAWNet
    """

    def __init__(self, vocab_size, embed_units, hidden_size,
                 padding_idx=None, num_layers=1, bidirectional=True,
                 attn_mode='mlp', dropout=0.0,
                 use_gpu=False):
        super(DAWNet, self).__init__()

        self.vocab_size = vocab_size
        self.embed_units = embed_units
        self.padding_idx = padding_idx

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.attn_mode = attn_mode
        self.use_gpu = use_gpu
