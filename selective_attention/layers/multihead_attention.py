# author: GAO Chenxi
# date: 2024/5/1 11:15
# -*- python version：3.8.10 -*-
# -*- coding: utf-8 -*-

import tensorflow as tf

class MultiHeadAttention(tf.Module):

    def __init__(self,
                 weights,
                 name=None,
                 **kwargs):
        self._weights = weights
        self._name = name
        super(MultiHeadAttention, self).__init__()

    '''
    the weight matrix shape is [num_heads, d_q/d_k/d_v, batch_size]
    decoder_input/encoder_output shape is [batch_size, height, width, channels]
    '''



