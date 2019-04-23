# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-23 21:06:36
# @Last Modified by:   liwei
# @Last Modified time: 2019-04-23 21:06:40

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))