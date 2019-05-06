# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-05-05 11:35:37
# @Last Modified by:   liwei
# @Last Modified time: 2019-05-05 21:11:26

from source.inputters.corpus import Corpus

from source.inputters.field import tokenize
from source.inputters.field import TextField


class HieraSrcCorpus(Corpus):
    """
    HieraSrcCorpus
    """

    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None,
                 min_len=0,
                 max_len=100,
                 embed_file=None,
                 share_vocab=False):
        super(HieraSrcCorpus, self).__init__(data_dir=data_dir,
                                             data_prefix=data_prefix,
                                             min_freq=min_freq,
                                             max_vocab_size=max_vocab_size)
        self.min_len = min_len
        self.max_len = max_len
        self.share_vocab = share_vocab

        self.SRC = TextField(tokenize_fn=tokenize,
                             embed_file=embed_file)
        if self.share_vocab:
            self.TGT = self.SRC
            self.CUE = self.SRC
        else:
            self.TGT = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)
            self.CUE = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)

        self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE}

    def read_data(self, data_file, data_type="train"):
        """
        read_data
        """
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                src, tgt, knowledge = line.strip().split('\t')[:3]
                filter_knowledge = []
                for sent in knowledge.split(''):
                    filter_knowledge.append(
                        ' '.join(sent.split()[:self.max_len]))

                filter_src = []
                for sent in src.split(':'):
                    filter_src.append(
                        ' '.join(sent.split()[:self.max_len]))
                data.append({'src': filter_src, 'tgt': tgt,
                             'cue': filter_knowledge})

        return data
