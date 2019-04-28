# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-23 21:04:32
# @Last Modified by:   liwei
# @Last Modified time: 2019-04-27 21:57:25

import torch
import torch.nn as nn

from source.modules.embedder import Embedder
from source.modules.attr import AttrProxy
from source.utils.misc import sequence_mask
from source.modules.attention import Attention
from source.modules.decoders.state import DecoderState
from source.utils.misc import Pack


class DecoderMemNN(nn.Module):

    def __init__(self, vocab, embedding_dim, hidden_size, hop,
                 dropout=0.0, num_layers=1, padding_idx=None, attn_mode=None):
        super(DecoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.attn_mode = attn_mode

        self.rnn_input_size = self.embedding_dim
        self.out_input_size = self.hidden_size

        for hop in range(self.max_hops + 1):
            C = Embedder(self.num_vocab, embedding_dim,
                         padding_idx=self.padding_idx)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")

        if self.attn_mode is not None:
            self.attention = Attention(query_size=self.hidden_size,
                                       memory_size=self.hidden_size,
                                       hidden_size=self.hidden_size,
                                       mode=self.attn_mode,
                                       project=False)
            self.rnn_input_size += self.hidden_size
            self.out_input_size += self.hidden_size

        self.softmax = nn.Softmax(dim=1)
        self.W = nn.Linear(self.embedding_dim, 1)
        self.W1 = nn.Linear(2 * self.embedding_dim, self.num_vocab)
        self.gru = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.embedding_dim,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)

    def initialize_state(self,
                         hidden,
                         attn_memory=None,
                         attn_mask=None,
                         memory_lengths=None):
        """
        initialize_state
        """
        if self.attn_mode is not None:
            assert attn_memory is not None

        if memory_lengths is not None and attn_mask is None:
            max_len = attn_memory.size(1)
            attn_mask = sequence_mask(memory_lengths, max_len).eq(0)

        init_state = DecoderState(
            hidden=hidden,
            attn_memory=attn_memory,
            attn_mask=attn_mask,
        )
        return init_state

    def load_memory(self, story):
        story_size = story.size()  # b * m * 3
        self.m_story = []
        for hop in range(self.max_hops):
            # .long()) # b * (m * s) * e
            embed_A = self.C[hop](
                story.contiguous().view(story.size(0), -1).long())
            embed_A = embed_A.view(
                story_size + (embed_A.size(-1),))  # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
            m_A = embed_A
            embed_C = self.C[
                hop + 1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            m_C = embed_C
            self.m_story.append(m_A)
        self.m_story.append(m_C)

    def decode(self, input, state, is_training=False):
        last_hidden = state.hidden
        rnn_input_list = []
        output = Pack()

        embed_q = self.C[0](input).unsqueeze(1)  # b * e --> b * 1 * e
        rnn_input_list.append(embed_q)

        if self.attn_mode is not None:
            attn_memory = state.attn_memory
            attn_mask = state.attn_mask
            query = last_hidden[-1].unsqueeze(1)
            weighted_context, attn = self.attention(query=query,
                                                    memory=attn_memory,
                                                    mask=attn_mask)
            rnn_input_list.append(weighted_context)
            output.add(attn=attn)

        rnn_input = torch.cat(rnn_input_list, dim=-1)
        output, new_hidden = self.gru(embed_q.unsqueeze(0), last_hidden)
        state.hidden = new_hidden

        u = [new_hidden[0].squeeze()]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if(len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)  # used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_lg = torch.sum(m_A * u_temp, 2)
            prob_p = self.softmax(prob_lg)
            m_C = self.m_story[hop + 1]

            prob = prob_p.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            if (hop == 0):
                p_vocab = self.W1(torch.cat((u[0], o_k), 1))
                prob_v = self.softmax(p_vocab)
            u_k = u[-1] + o_k
            u.append(u_k)
        p_ptr = prob_lg

        if is_training:
            # p_ptr, p_vocab 是 softmax 之前的值， 不是概率
            return p_ptr, p_vocab, state, output
        else:
            return prob_p, prob_v, state, output

    def forward(self, inputs, state):
        """
        forward
        """
        inputs, lengths = inputs
        batch_size, max_len = inputs.size()

        out_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.out_input_size),
            dtype=torch.float)

        # sort by lengths
        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)
        state = state.index_select(indices)

        # number of valid input (i.e. not padding index) in each time step
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)

        for i, num_valid in enumerate(num_valid_list):
            dec_input = inputs[:num_valid, i]
            valid_state = state.slice_select(num_valid)
            out_input, valid_state, _ = self.decode(
                dec_input, valid_state, is_training=True)
            state.hidden[:, :num_valid] = valid_state.hidden
            out_inputs[:num_valid, i] = out_input.squeeze(1)

        # Resort
        _, inv_indices = indices.sort()
        state = state.index_select(inv_indices)
        out_inputs = out_inputs.index_select(0, inv_indices)

        log_probs = self.output_layer(out_inputs)
        return log_probs, state
