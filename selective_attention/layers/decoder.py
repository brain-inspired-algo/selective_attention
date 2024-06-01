# author: GAO Chenxi
# date: 2024/4/26 13:43
# -*- python version：3.8.10 -*-
# -*- coding: utf-8 -*-

import tensorflow as tf
import selective_attention.utils.common_utils as utils
import tensorflow.estimator.Modekeys as MODES


class Decoder(tf.Module):
    def __init__(self,
                 hparams,
                 name=None,
                 **kwargs):
        self._hparams = hparams
        self._name = name
        super(Decoder, self).__init__()

    '''
    decoder_inputs shape: (batch_size, 1, hiddensize)
    '''

    def __call__(self, decoder_inputs, encoder_output=None, num_heads=8, mask=None):
        if encoder_output == None:

