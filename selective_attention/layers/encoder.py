# author: GAO Chenxi
# date: 2024/4/26 13:42
# -*- python version：3.8.10 -*-
# -*- coding: utf-8 -*-
import tensorflow as tf


class Encoder(tf.Module):
    def __init__(self,
                 weights,
                 name=None,
                 **kwargs):
        self._weights = weights
        self._name = name


        super(Encoder, self).__init__()

    '''
    input shape:(batch_size, image_len, image_len, num_channels)
    use the weights to convert q, k, v
    '''

    def __call__(self, inputs, num_heads=8, training=True):
