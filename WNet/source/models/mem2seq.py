# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-23 20:59:04
# @Last Modified by:   liwei
# @Last Modified time: 2019-05-08 10:17:01

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.modules.encoders.mem_encoder import EncoderMemNN
from source.modules.decoders.mem_decoder import DecoderMemNN
from source.modules.encoders.rnn_encoder import RNNEncoder
from source.utils.criterions import NLLLoss
from source.utils.metrics import accuracy
from source.utils.misc import Pack


class Mem2Seq(BaseModel):

    def __init__(self, vocab_size, hidden_size, embed_units, max_hop=3,
                 padding_idx=None, n_layers=1, attn_mode='mlp',
                 dropout=0.0, use_gpu=False):
        super(Mem2Seq, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_units = embed_units
        self.max_hop = max_hop
        self.padding_idx = padding_idx
        self.attn_mode = attn_mode
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.n_layers = n_layers

        enc_embedder = Embedder(num_embeddings=self.vocab_size,
                                embedding_dim=self.embed_units,
                                padding_idx=self.padding_idx)

        self.rnn_encoder = RNNEncoder(input_size=self.embed_units,
                                      hidden_size=self.hidden_size,
                                      embedder=enc_embedder,
                                      dropout=self.dropout)

        self.mem_encoder = EncoderMemNN(vocab=self.vocab_size,
                                        hidden_size=self.hidden_size,
                                        hop=self.max_hop,
                                        attn_mode='general',
                                        padding_idx=self.padding_idx)

        self.decoder = DecoderMemNN(vocab=self.vocab_size,
                                    embedding_dim=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    hop=self.max_hop,
                                    dropout=self.dropout,
                                    padding_idx=self.padding_idx)

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
        enc_inputs = _, lengths = inputs.src[0][:, 1:-1], inputs.src[1] - 2
        enc_outputs, enc_hidden = self.rnn_encoder(enc_inputs, hidden)

        # knowledge
        batch_size, sent_num, sent = inputs.cue[0].size()
        tmp_len = inputs.cue[1]
        tmp_len[tmp_len > 0] -= 2
        cue_inputs = inputs.cue[0][:, :, 1:-1], tmp_len

        self.decoder.load_memory(inputs.cue[0][:, :, 1:-1])

        u = self.mem_encoder(cue_inputs, enc_hidden[-1])

        dec_init_state = self.decoder.initialize_state(
            hidden=u.unsqueeze(0),
            cue_inputs=inputs.cue[0][:, :, 1:-1],
            attn_memory=enc_outputs if self.attn_mode else None,
            memory_lengths=lengths if self.attn_mode else None)

        return outputs, dec_init_state

    def decode(self, input, state):
        """
        decode
        """
        log_prob, state, output = self.decoder.decode(input, state)
        return log_prob, state, output

    def forward(self, enc_inputs, dec_inputs, hidden=None):
        """
        forward
        """
        outputs, dec_init_state = self.encode(enc_inputs, hidden)

        p_vocab, state = self.decoder(dec_inputs, dec_init_state)
        outputs.add(logits=p_vocab)
        return outputs

    def collect_metrics(self, outputs, target):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        logits = outputs.logits
        scores = -self.nll_loss(logits, target, reduction=False)
        nll = self.nll_loss(logits, target)
        num_words = target.ne(self.padding_idx).sum().item()
        acc = accuracy(logits, target, padding_idx=self.padding_idx)
        metrics.add(nll=(nll, num_words), acc=acc)
        loss += nll

        metrics.add(loss=loss)
        return metrics, scores

    def iterate(self, inputs, optimizer=None, grad_clip=None,
                is_training=False, epoch=-1):
        """
        iterate
        """
        enc_inputs = inputs
        dec_inputs = inputs.tgt[0][:, :-1], inputs.tgt[1] - 1
        target = inputs.tgt[0][:, 1:]

        outputs = self.forward(enc_inputs, dec_inputs)
        metrics, scores = self.collect_metrics(outputs, target)

        loss = metrics.loss
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        if is_training:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(parameters=self.parameters(),
                                max_norm=grad_clip)
            optimizer.step()
        return metrics, scores
