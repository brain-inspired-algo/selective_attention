# author: GAO Chenxi
# date: 2024/4/26 14:48
# -*- python version：3.8.10 -*-
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.estimator.ModeKeys as MODES
import selective_attention.utils.common_utils as utils

'''
recognition target input shape is [batch_size, length, width,channels]
for categorical problem input label shape is [batch_size, 1]
inference shape is [batch_size, inference_len, 1, 1]
'''
def prepare_decoding(inputs, mode, hparams):

    labels = inputs
    if mode == MODES.PREDICT:
        labels_shape = utils.shape_list(labels)
        inference_len = labels_shape[1]
        assert hparams.image_len * hparams.channels % hparams.query_shape[1] == 0
        assert hparams.image_len % hparams.query_shape[0] == 0
        total_block_len = hparams.image_len * hparams.channels
        block_padding_factor = total_block_len * hparams.query_shape[0]
        labels = tf.pad(inputs, [
            [0, 0], [0, -inference_len % block_padding_factor],
            [0, 0], [0, 0]])

        num_blocks = total_block_len // hparams.query_shape[1]
        # Reshape the image to represent blocks
        target_blocks = tf.reshape(
            labels, [labels_shape[0], -1, num_blocks, hparams.query_shape[0],
                     hparams.query_shape[1]])
        # Transpose to read the image in 2D fashion.
        labels = tf.transpose(target_blocks, [0, 1, 3, 2, 4])
        labels = tf.reshape(labels, [labels_shape[0], -1, hparams.image_len, hparams.channels])

    decoder_inputs = utils.right_shift_blockwise(labels, hparams.query_shape, name='dec_inputs')
    decoder_inputs = utils.add_pos_signals(decoder_inputs, hparams._original_hparams)
    decoder_inputs = utils.cast_like(decoder_inputs, inputs)
    return decoder_inputs

'''
if the context is appropriate to generate summaries
'''
def should_generate_summaries():

  name_scope = tf.get_current_name_scope()
  if name_scope and "while/" in name_scope:
    # Summaries don't work well within tf.while_loop()
    return False
  if tf.get_variable_scope().reuse:
    # Avoid generating separate summaries for different data shards
    return False
  return True

'''
expand and squeeze dimensions to make n-d x
'''
def expand_squeeze_to_nd(x, n, squeeze_dim=2, expand_dim=-1):

  if len(x.shape) > n:
    while len(x.shape) != n:
      x = tf.squeeze(x, [squeeze_dim])
  else:
    while len(x.shape) != n:
      x = tf.expand_dims(x, expand_dim)
  return x

'''
get list of dims
'''
def shape_list(x):

  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, dim in enumerate(static):
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

'''
cast x to y's dtype
'''
def cast_like(x, y):

  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)

  if x.dtype.base_dtype == y.dtype.base_dtype:
    return x

  cast_x = tf.cast(x, y.dtype)
  if cast_x.device != x.device:
    x_name = "(eager Tensor)"
    try:
      x_name = x.name
    except AttributeError:
      pass
    tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                       x.device, cast_x.device)
  return cast_x

'''
right shifts a query_shape in every block
'''
def right_shift_blockwise(x, query_shape, name=None):

  with tf.variable_scope(
      name, default_name="right_shift_blockwise", values=[x]):
    if type(x) == np.ndarray:
        x_list_shape = list(x.shape)
    else:
        x_list_shape = x.get_shape().as_list()
    x_shape = utils.shape_list(x)
    # Add a dummy dimension for heads.
    x = tf.expand_dims(x, axis=1)
    x = pad_to_multiple_2d(x, query_shape)
    padded_x_shape = common_layers.shape_list(x)
    # Set up q blocks.
    x_indices = gather_indices_2d(x, query_shape, query_shape)
    x_new = get_shifted_center_blocks(x, x_indices)

    # Put representations back into original shapes.
    output = scatter_blocks_2d(x_new, x_indices, padded_x_shape)

    # Remove the dummy head dimension.
    known_axes = [i for i, size in enumerate(output.get_shape()) if size == 1]
    output = tf.squeeze(output, axis = known_axes)
    output = tf.transpose(output, perm =[0, 2, 1])

    # Remove the padding if introduced.
    begin = tf.convert_to_tensor([0, 0, 0], dtype=tf.int32)
    size = tf.convert_to_tensor([x_shape[0], x_shape[1], x_shape[2]], dtype=tf.int32)
    output = tf.slice(output, begin, size)
    output.set_shape(x_list_shape)
    return output

'''
add embeddings to the input tensor
'''
def add_pos_signals(x, hparams, name="pos_emb"):
  with tf.variable_scope(name, reuse=False):
    if hparams.pos == "timing":
      x = add_timing_signal_nd(x)
    else:
      assert hparams.pos == "emb"
      x = add_positional_embedding_nd(
          x, hparams.max_length, name)
  return x