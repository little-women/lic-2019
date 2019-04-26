# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-23 21:03:41
# @Last Modified by:   liwei
# @Last Modified time: 2019-04-26 15:35:30

import torch
import torch.nn as nn

from source.modules.embedder import Embedder
from source.modules.attention import Attention
from source.modules.attr import AttrProxy


class EncoderMemNN(nn.Module):

    def __init__(self, vocab, hidden_size, hop=1,
                 attn_mode='dot', padding_idx=None):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.hidden_size = hidden_size
        self.attn_mode = attn_mode
        self.padding_idx = padding_idx

        for hop in range(self.max_hops + 1):
            C = Embedder(self.num_vocab, self.hidden_size,
                         padding_idx=self.padding_idx)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)

        for hop in range(self.max_hops):
            A = Attention(query_size=self.hidden_size,
                          memory_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          mode=self.attn_mode,
                          return_attn_only=True)
            self.add_module("A_{}".format(hop), A)

        self.C = AttrProxy(self, "C_")
        self.A = AttrProxy(self, "A_")
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, enc_hidden):
        """
        enc_hidden: batch_size, query_size
        inputs: batch_size, memory_length, max_len
        lengths: batch_size, memory_length

        Return: batch_size, memory_size

        """
        if isinstance(inputs, tuple):
            inputs, lengths = inputs
        else:
            inputs, lengths = inputs, None

        u = [enc_hidden]
        attns = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](
                inputs.contiguous().view(inputs.size(0), -1).long())
            embed_A = embed_A.view(inputs.size() + (embed_A.size(-1),))
            # batch_size, memory_length, embedding_dim
            m_A = torch.sum(embed_A, 2).squeeze(2)

            attn = self.A[hop](query=u[-1].unsqueeze(1),
                               memory=m_A, mask=lengths.eq(0))

            attns.append(attn)

            embed_C = self.C[
                hop + 1](inputs.contiguous().view(inputs.size(0), -1).long())
            embed_C = embed_C.view(inputs.size() + (embed_C.size(-1),))
            m_C = torch.sum(embed_C, 2).squeeze(2)

            o_k = torch.bmm(attn, m_C).squeeze(1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return u_k, attns
