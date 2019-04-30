# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-30 10:35:53
# @Last Modified by:   liwei
# @Last Modified time: 2019-04-30 23:28:32

import torch
import torch.nn as nn

from source.modules.attention import Attention
from source.modules.decoders.state import DecoderState
from source.utils.misc import Pack
from source.utils.misc import sequence_mask
from source.utils.misc import convert_dist


class PointerDecoder(nn.Module):
    """
    A GRU recurrent neural network decoder.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 embedder=None,
                 num_layers=1,
                 attn_mode='mlp',
                 dropout=0.0):
        super(PointerDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.attn_mode = None if attn_mode == 'none' else attn_mode
        self.dropout = dropout

        self.memory_size = hidden_size

        self.rnn_input_size = self.input_size
        self.out_input_size = self.hidden_size + self.input_size

        if self.attn_mode is not None:
            self.hist_attention = Attention(query_size=self.hidden_size,
                                            memory_size=self.hidden_size,
                                            hidden_size=self.hidden_size,
                                            mode=self.attn_mode,
                                            project=False)
            self.fact_attention = Attention(query_size=self.hidden_size,
                                            memory_size=self.hidden_size,
                                            hidden_size=self.hidden_size,
                                            mode=self.attn_mode,
                                            project=False)
            self.rnn_input_size += self.memory_size * 2
            self.out_input_size += self.memory_size * 2

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(self.out_input_size, 3),
            nn.Softmax(dim=-1)
        )

        if self.out_input_size > self.hidden_size:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.output_size),
                nn.Softmax(dim=-1),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.output_size),
                nn.Softmax(dim=-1),
            )

    def initialize_state(self,
                         hidden,
                         fact,
                         hist,
                         attn_fact=None,
                         attn_hist=None,
                         hist_mask=None,
                         fact_mask=None,
                         fact_lengths=None,
                         hist_lengths=None):
        """
        initialize_state
        """
        if self.attn_mode is not None:
            assert attn_fact is not None
            assert attn_hist is not None

        if hist_lengths is not None and hist_mask is None:
            max_len = attn_hist.size(1)
            hist_mask = sequence_mask(hist_lengths, max_len).eq(0)

        if fact_lengths is not None and fact_mask is None:
            sent_len = torch.max(fact_lengths)
            fact_mask = sequence_mask(fact_lengths, sent_len).eq(
                0).view(fact_lengths.size(0), -1)

        init_state = DecoderState(
            hidden=hidden,
            fact=fact,
            hist=hist,
            attn_hist=attn_hist,
            attn_fact=attn_fact,
            hist_mask=hist_mask,
            fact_mask=fact_mask
        )
        return init_state

    def decode(self, input, state, is_training=False):
        """
        decode
        """
        hidden = state.hidden
        rnn_input_list = []
        out_input_list = []
        output = Pack()

        if self.embedder is not None:
            input = self.embedder(input)

        # shape: (batch_size, 1, input_size)
        input = input.unsqueeze(1)

        rnn_input_list.append(input)
        out_input_list.append(input)

        if self.attn_mode is not None:
            # (batch_size, 1, hidden_size)
            query = hidden[-1].unsqueeze(1)

            # history attention
            weighted_hist, attn_h = self.hist_attention(query=query,
                                                        memory=state.attn_hist,
                                                        mask=state.hist_mask)
            rnn_input_list.append(weighted_hist)
            out_input_list.append(weighted_hist)
            output.add(attn_h=attn_h)

            # fact attention
            weighted_fact, attn_f = self.fact_attention(query=query,
                                                        memory=state.attn_fact,
                                                        mask=state.fact_mask)
            rnn_input_list.append(weighted_fact)
            out_input_list.append(weighted_fact)
            output.add(attn_f=attn_f)

        rnn_input = torch.cat(rnn_input_list, dim=-1)
        rnn_output, new_hidden = self.rnn(rnn_input, hidden)
        out_input_list.append(rnn_output)

        # cat (fact_hidden, hist_hidden, hidden, x)
        # (batch_size, 1, out_input_size)
        out_input = torch.cat(out_input_list, dim=-1)
        state.hidden = new_hidden

        if is_training:
            return out_input, state, output
        else:
            p_mode = self.ff(out_input)

            # prob_hist = input.new_zeros(
            #     size=(batch_size, 1, self.output_size),
            #     dtype=torch.float)

            # prob_fact = input.new_zeros(
            #     size=(batch_size, 1, self.output_size),
            #     dtype=torch.float)

            prob_vocab = self.output_layer(out_input)

            weighted_prob = prob_vocab * p_mode[:, :, 0].unsqueeze(2)
            weighted_f = output.attn_f * p_mode[:, :, 1].unsqueeze(2)
            weighted_h = output.attn_h * p_mode[:, :, 2].unsqueeze(2)
            weighted_prob = convert_dist(
                weighted_h, state.hist, weighted_prob)
            weighted_prob = convert_dist(
                weighted_f, state.fact, weighted_prob)

            # a = torch.cat((prob_vocab, prob_hist, prob_fact), -
            #               1).view(batch_size * 1, self.output_size, -1)
            # b = p_mode.view(batch_size * 1, -1).unsqueeze(2)

            # prob = torch.bmm(a, b).squeeze().view(batch_size, 1, -1)

            log_prob = torch.log(weighted_prob + 1e-10)
            return log_prob, state, output

    def forward(self, inputs, state):
        """
        forward
        """
        inputs, lengths = inputs
        batch_size, max_len = inputs.size()

        out_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.out_input_size),
            dtype=torch.float)

        fact_len = state.fact.size(1)
        hist_len = state.hist.size(1)
        out_facts = inputs.new_zeros(
            size=(batch_size, max_len, fact_len),
            dtype=torch.float)
        out_hists = inputs.new_zeros(
            size=(batch_size, max_len, hist_len),
            dtype=torch.float)

        # prob_hist = inputs.new_zeros(
        #     size=(batch_size, max_len, self.output_size),
        #     dtype=torch.float)

        # prob_fact = inputs.new_zeros(
        #     size=(batch_size, max_len, self.output_size),
        #     dtype=torch.float)

        # sort by lengths
        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)
        state = state.index_select(indices)

        # number of valid input (i.e. not padding index) in each time step
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)

        for i, num_valid in enumerate(num_valid_list):
            dec_input = inputs[:num_valid, i]
            valid_state = state.slice_select(num_valid)
            out_input, valid_state, output = self.decode(
                dec_input, valid_state, is_training=True)
            state.hidden[:, :num_valid] = valid_state.hidden
            out_inputs[:num_valid, i] = out_input.squeeze(1)
            out_facts[:num_valid, i] = output.attn_f.squeeze(1)
            out_hists[:num_valid, i] = output.attn_h.squeeze(1)

        # Resort
        _, inv_indices = indices.sort()
        state = state.index_select(inv_indices)
        out_inputs = out_inputs.index_select(0, inv_indices)
        out_facts = out_facts.index_select(0, inv_indices)
        out_hists = out_hists.index_select(0, inv_indices)

        p_modes = self.ff(out_inputs)

        # (batch_size, max_len, vocab_size)
        prob_vocab = self.output_layer(out_inputs)
        # prob_hist = convert_dist(
        #     out_hists, state.hist, prob_hist)
        # prob_fact = convert_dist(
        #     out_facts, state.fact, prob_fact)

        # a = torch.cat((prob_vocab, prob_hist, prob_fact), -
        #               1).view(batch_size * max_len, self.output_size, -1)
        # b = p_modes.view(batch_size * max_len, -1).unsqueeze(2)
        # prob = torch.bmm(a, b).squeeze().view(batch_size, max_len, -1)

        weighted_prob = prob_vocab * p_modes[:, :, 0].unsqueeze(2)
        weighted_f = out_facts * p_modes[:, :, 1].unsqueeze(2)
        weighted_h = out_hists * p_modes[:, :, 2].unsqueeze(2)
        weighted_prob = convert_dist(
            weighted_h, state.hist, weighted_prob)
        weighted_prob = convert_dist(
            weighted_f, state.fact, weighted_prob)

        log_probs = torch.log(weighted_prob + 1e-10)
        return log_probs, state, output
