# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-05-13 20:30:13
# @Last Modified by:   liwei
# @Last Modified time: 2019-05-13 20:33:54

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from torch.nn.utils import clip_grad_norm_

from source.models.base_model import BaseModel
from source.modules.context_rnn import ContextRNN
from source.modules.context_rnn import ExternalKnowledge
from source.modules.context_rnn import LocalMemoryDecoder


class GLMP(BaseModel):

    def __init__(self, hidden_size, lang, max_resp_len, path, task, lr,
                 n_layers, dropout, use_gpu=False):
        super(GLMP, self).__init__()
        self.name = "GLMP"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.max_resp_len = max_resp_len
        self.decoder_hop = n_layers
        self.softmax = nn.Softmax(dim=0)

        if path:
            if self.use_gpu:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th')
                self.extKnow = torch.load(str(path) + '/enc_kb.th')
                self.decoder = torch.load(str(path) + '/dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(
                    str(path) + '/enc.th', lambda storage, loc: storage)
                self.extKnow = torch.load(
                    str(path) + '/enc_kb.th', lambda storage, loc: storage)
                self.decoder = torch.load(
                    str(path) + '/dec.th', lambda storage, loc: storage)
        else:
            self.encoder = ContextRNN(lang.n_words, hidden_size, dropout)
            self.extKnow = ExternalKnowledge(
                lang.n_words, hidden_size, n_layers, dropout)
            # Generator(lang, hidden_size, dropout)
            self.decoder = LocalMemoryDecoder(self.encoder.embedding,
                                              lang, hidden_size,
                                              self.decoder_hop, dropout)

        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.extKnow_optimizer = optim.Adam(self.extKnow.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.decoder_optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        self.criterion_bce = nn.BCELoss()
        self.reset()

        if use_gpu:
            self.encoder.cuda()
            self.extKnow.cuda()
            self.decoder.cuda()
