# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-24 10:29:03
# @Last Modified by:   liwei
# @Last Modified time: 2019-04-25 12:10:37

import argparse

from source.models.memnet import *
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
        model = globals()[config.model]()
    elif config.model == 'MemNet':
        model = globals()[config.model](10000, 300, 800)
    elif config.model == 'EncoderMemNN':
        model = globals()[config.model](10000, 100, 2)

        enc_hidden = torch.rand(3, 100)
        lengths = torch.tensor([[2, 3, 2, 0, 0, 0, 0, 0, 0, 0], [
                               2, 3, 2, 4, 5, 0, 0, 0, 0, 0], [1, 3, 2, 5, 0, 0, 0, 0, 0, 0]])
        story = torch.ones(3, 10, 7)
        u = model.forward((story, lengths), enc_hidden)
        print(u)
    elif config.model == 'DecoderMemNN':
        model = globals()[config.model](10000, 300, 3, 0.2, False, 1)
        story = torch.ones(3, 10, 7)
        model.load_memory(story)
        for m in model.m_story:
            print(m.size())

        decoder_input = Variable(torch.LongTensor([2] * 3))
    print(model)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
