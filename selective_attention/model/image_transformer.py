# author: GAO Chenxi
# date: 2024/4/26 13:42
# -*- python version：3.8.10 -*-
# -*- coding: utf-8 -*-

import copy
import tensorflow as tf
from selective_attention.utils.attention_model import AttentionModel
import selective_attention.utils.common_utils as utils
import tensorflow.estimator.ModeKeys as MODES
from selective_attention.layers.decoder import prepare_decoder

class ImageTransformer(AttentionModel):

    def model(self, features):
        hparams = copy.copy(self._hparams)
        inputs = features['inputs']
        targets = features['targets']
        targets_shape = utils.shape_list(targets)
        if not (tf.get_variable_scope().reuse or hparams.mode == MODES.PREDICT):
            tf.summary.image("targets", targets, max_outputs=1)

        # prepare decoder inputs and bias
        decoder_input, rows, cols = prepare_decoder(
            inputs, hparams)

        # Extra losses list if we want to use moe.
        losses = []

        # Add class label to decoder input.
        if not hparams.unconditional:
            decoder_input += tf.reshape(
                inputs,
                [utils.shape_list(targets)[0], 1, 1, hparams.hidden_size])
        decoder_output = transformer_decoder_layers(
            decoder_input,
            None,
            hparams.num_decoder_layers or hparams.num_hidden_layers,
            hparams,
            attention_type=hparams.dec_attention_type,
            losses=losses,
            name="decoder")
        output = create_output(decoder_output, rows, cols, targets, hparams)

        if losses:
            print("extra loss output")
            return output, {"extra_loss": tf.add_n(losses)}
        else:
            return output