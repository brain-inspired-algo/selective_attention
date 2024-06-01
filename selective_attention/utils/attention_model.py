# author: GAO Chenxi
# date: 2024/4/26 13:48
# -*- python version：3.8.10 -*-
# -*- coding: utf-8 -*-
import six

import tensorflow as tf
from tensorflow.estimator import ModeKeys as MODES

from selective_attention.layers.encoder import Encoder
from selective_attention.layers.decoder import Decoder
import selective_attention.utils.common_utils as utils

class AttentionModel(tf.Module):

    batch_size = 64
    hidden_size = 64
    num_attention_encoder = 6
    num_attention_decoder = 6
    query_shape = (4, 4)
    image_len = 28
    channels = 1
    mode = MODES.TRAIN
    loss_history = []
    weights = None
    num_heads = 8


    def __init__(self,
                 hparams,
                 name = None,
                 mode = MODES.TRAIN,
                 **kwargs):
        super(AttentionModel, self).__init__(trainable = mode , name = name,  **kwargs)

        '''copy hparams...'''
        if hparams.hidden_size:
            self.hidden_size = hparams.hidden_size

        self._original_hparams = hparams
        self.mode = mode

    def __call__(self, inputs, labels):

        encoder_inputs = utils.add_pos_signals(inputs, self._original_hparams)

        for layer in range(self.num_attention_encoder):
            name = 'encoder_{}'.format(layer)
            encoder = Encoder(self.weights, name=name)
            encoder_inputs = encoder(encoder_inputs, self.num_heads)

        decoder_inputs = utils.prepare_decoding(labels, self.mode, self._original_hparams)

        for layer in range(self.num_attention_decoder):
            name = 'decoder_{}'.format(layer)
            decoder = Decoder(self._original_hparams, name=name)
            decoder_inputs = decoder(encoder_inputs, decoder_inputs)

        l_outputs = tf.nn.linear(inputs, self._original_hparams)
        s_outputs = tf.nn.softmax(l_outputs)

        return s_outputs

    def loss(self, labels, s_outputs):

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=s_outputs)

        return loss

    def optimizer(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = optimizer.minimize(loss)
        return optimizer

    '''demo which could be implemented in the training script'''
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.__call__(images, labels)
            loss_value = self.loss(labels, predictions).numpy().mean()
        self.loss_history.append(loss_value)
        self.optimizer(loss_value, 0.01)




    '''
    setting the logger
    '''

    _already_logged = set()

    def _eager_log(self, level, *args):
        if tf.executing_eagerly() and args in self._already_logged:
            return
        self._already_logged.add(args)
        getattr(tf.get_logger(), level)(*args)

    def log_debug(self, *args):
        self._eager_log("debug", *args)

    def log_info(self, *args):
        self._eager_log("info", *args)

    def log_warn(self, *args):
        self._eager_log("warn", *args)

    def set_mode(self, mode):
        self.log_info("Setting Mode to '%s'", mode)
        hparams = self._original_hparams
        hparams.add_hparam("mode", mode)
        if mode != tf.estimator.ModeKeys.TRAIN:
            for key in hparams.values():
                if key.endswith("dropout") or key == "label_smoothing":
                    self.log_info("Setting hparams.%s to 0.0", key)
                    setattr(hparams, key, 0.0)
        self._hparams = hparams

