# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-24 10:29:03
# @Last Modified by:   liwei
# @Last Modified time: 2019-04-27 17:00:10

import argparse
from torch.autograd import Variable

from source.models.memnet import *
from source.models.mem2seq import *
from source.modules.encoders.mem_encoder import EncoderMemNN
from source.modules.decoders.mem_decoder import DecoderMemNN


def model_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, default='MemNet')

    config = parser.parse_args()

    return config


def main():
    global model
    config = model_config()
    if config.model == 'Mem2Seq':
        model = globals()[config.model](10000, 300)
    elif config.model == 'MemNet':
        model = globals()[config.model](10000, 300, 800)
    elif config.model == 'EncoderMemNN':
        model = globals()[config.model](10000, 100, 2)

        enc_hidden = torch.rand(3, 100)
        lengths = torch.tensor([[2, 3, 2, 0, 0, 0, 0, 0, 0, 0], [
                               2, 3, 2, 4, 5, 0, 0, 0, 0, 0], [1, 3, 2, 5, 0, 0, 0, 0, 0, 0]])
        story = torch.ones(3, 10, 7)
        u, attns = model.forward((story, lengths), enc_hidden)
        print(attns)
    elif config.model == 'DecoderMemNN':
        model = globals()[config.model](10000, 300, 3, 0.2, False, 1)
        story = torch.ones(3, 10, 7)
        model.load_memory(story)
        for m in model.m_story:
            print(m.size())

        decoder_input = Variable(torch.LongTensor([2] * 3))
        last_hidden = Variable(torch.rand(1, 3, 300))
        p_ptr, p_vocab, hidden = model.ptrMemDecoder(decoder_input, last_hidden)
        print(p_ptr)
        print(p_vocab)

    print(model)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
