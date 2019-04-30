# -*- coding: utf-8 -*-
# @Author: Wei Li
# @Date:   2019-04-30 18:30:33
# @Last Modified by:   liwei
# @Last Modified time: 2019-04-30 18:32:46


from source.models.base_model import BaseModel


class DAWNet(BaseModel):
    """
    docstring for DAWNet
    """

    def __init__(self, arg):
        super(DAWNet, self).__init__()
        self.arg = arg
