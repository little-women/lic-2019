# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-05-08 14:42:31
# @Last Modified by:   liwei
# @Last Modified time: 2019-05-08 17:28:08

from source.inputters.corpus import Corpus

from source.inputters.field import tokenize
from source.inputters.field import TextField
from source.inputters.field import NumberField


class TriSrcCorpus(Corpus):
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
        super(TriSrcCorpus, self).__init__(data_dir=data_dir,
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

        self.TAG = NumberField()

        self.fields = {'src': self.SRC, 'tgt': self.TGT,
                       'cue': self.CUE, 'tag': self.TAG}

    def genTag(self, tgt, knowledge):
        tag = []
        entities = []
        for k in knowledge:
            entities.extend(k.split())
        knowledge_set = set(entities)
        for t in tgt.split():
            if t in knowledge_set:
                tag.append(1)
            else:
                tag.append(0)
        return tag, knowledge_set

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

                tags, knowledge_set = self.genTag(tgt, filter_knowledge)

                data.append({'src': src, 'tgt': tgt,
                             'cue': filter_knowledge,
                             'tag': tags})

        return data
