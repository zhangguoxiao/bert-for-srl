# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import tensor_shape

def leaky_relu(x, alpha=0.2, max_value=None):
    '''ReLU.

    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float16),
                             tf.cast(max_value, dtype=tf.float16))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x


def BiLSTM(rnn_inputs, keep_prop, seq_lenths, hidden_size, time_major = False, return_outputs = True, type = 'concat'):
    '''
    构建模型
    :param data:placeholder
    :param FLAGS.mem_dim:
    :return:
    '''
    if not time_major:
        rnn_inputs  = tf.transpose(rnn_inputs, [1, 0, 2])   ##time major
    cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob=keep_prop)
    cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob=keep_prop)
    '''If time_major == True, this must be a Tensor of shape: [max_time, batch_size, ...], or a nested tuple of such elements.     '''
    (fw_outputs,bw_outputs), _  = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, sequence_length = seq_lenths,
                                                      dtype=tf.float32, time_major = True)
    if return_outputs:
        if type == 'concat':
            time_major_results = tf.concat((fw_outputs,bw_outputs), 2)
            return tf.transpose(time_major_results, [1, 0, 2])
        else:
            time_major_results = tf.add(fw_outputs, bw_outputs)
            return tf.transpose(time_major_results, [1, 0, 2])
    else:
        if type == 'concat':
            return tf.reduce_mean(tf.concat((fw_outputs,bw_outputs), 2), 0)
            #return tf.reduce_max(tf.concat((fw_outputs, bw_outputs), 2), 0)
        else:
            hidden = tf.add(fw_outputs, bw_outputs)
            return tf.reduce_max(hidden, axis=0)

def _reverse(input_, seq_lengths, seq_dim, batch_dim):
      if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
      else:
        return array_ops.reverse(input_, axis=[seq_dim])

def MultiLayerLSTM(inputs, lstm_keep_prop, seq_lenths, hidden_size, layers, time_major = False):
    if not time_major:
        inputs  = tf.transpose(inputs, [1, 0, 2])   ##time major

    def fn(input, name=None, reuse = False):
            return tf.layers.dense(input, units = hidden_size, name=name, reuse= reuse)
    hidden_0 = time_distributed([fn],inputs,['dense'+ str(0)])
    #hidden_0 = tf.layers.batch_normalization(hidden_0)
    #hidden_0 = tf.nn.dropout(hidden_0, lstm_keep_prop)
    with vs.variable_scope("fw_0") as fw_scope:
        #cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob=lstm_keep_prop)
        cell_fw = tf.contrib.rnn.LSTMCell(hidden_size)
        lstm_0, _ = tf.nn.dynamic_rnn(
              cell=cell_fw, inputs=hidden_0, sequence_length=seq_lenths, dtype=tf.float32, time_major = True)
        lstm_0 = tf.layers.batch_normalization(lstm_0)
        lstm_0 = tf.nn.dropout(lstm_0, lstm_keep_prop)
    input_tmp = tf.concat((hidden_0,lstm_0), 2)
    #input_tmp = tf.layers.batch_normalization(input_tmp)
    #input_tmp = tf.nn.dropout(input_tmp, keep_prop)
    for i in range(1, layers):
        def fn(input, name=None, reuse = False):
            return tf.layers.dense(input, units = hidden_size, name=name, reuse= reuse)
        fc = time_distributed([fn],input_tmp,['dense_'+ str(i)])
        #fc = tf.layers.batch_normalization(fc)
        #fc = tf.nn.dropout(fc, lstm_keep_prop)
        if i % 2 == 1:
             with vs.variable_scope("bw_" + str(i)) as bw_scope:
                    fc_reverse = _reverse( fc, seq_lengths=seq_lenths,seq_dim=0, batch_dim=1)
                    #cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob=lstm_keep_prop)
                    cell_bw = tf.contrib.rnn.LSTMCell(hidden_size)
                    temp, _ = tf.nn.dynamic_rnn(
                                cell=cell_bw, inputs=fc_reverse, sequence_length=seq_lenths, dtype=tf.float32, time_major = True)
                    lstm = _reverse(temp, seq_lengths=seq_lenths, seq_dim=0, batch_dim=1)

        else:
             with vs.variable_scope("fw_" + str(i)) as fw_scope:
                    #cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob=lstm_keep_prop)
                    cell_fw = tf.contrib.rnn.LSTMCell(hidden_size)
                    lstm, _ = tf.nn.dynamic_rnn(
                                cell=cell_fw, inputs=fc, sequence_length=seq_lenths, dtype=tf.float32, time_major = True)
        lstm = tf.layers.batch_normalization(lstm)
        lstm = tf.nn.dropout(lstm, lstm_keep_prop)
        input_tmp = tf.concat((fc, lstm), 2)
        #input_tmp = tf.layers.batch_normalization(input_tmp)
        #input_tmp = tf.nn.dropout(input_tmp, keep_prop)
    return tf.transpose(input_tmp, [1, 0, 2])

def DynamicBiaffine(input1, input2, out_size ,verbs_for_index, name,dim,batch_seqlenth,batch_size,
                 bias=(True, True, True)):
    dim1 = dim2 = dim
    if bias[0]:
            ones1 = tf.fill([tf.shape(input1)[0], tf.shape(input1)[1], 1], 1.0)   ##这里fill支持动态len1
            input1 = tf.concat([input1, ones1], axis=-1)
            dim1 += 1
    if bias[1]:
            ones2 = tf.fill([tf.shape(input2)[0], tf.shape(input2)[1], 1], 1.0)
            input2 = tf.concat([input2, ones2], axis=-1)
            dim2 += 1

    W = tf.get_variable(
                "W_bi_" + name,
                shape=[dim1, dim2 * out_size],
                dtype=tf.float32,
                initializer= tf.truncated_normal_initializer)
    input1_reshaped = tf.reshape(input1, [-1, dim1])
    affine = tf.matmul(input1_reshaped, W)
    affine = tf.reshape(affine, ([batch_size, -1, dim2]))
    biaffine   = tf.matmul(affine, tf.transpose(input2, (0,2,1)))
    biaffine = tf.reshape(biaffine, [batch_size, -1, out_size, batch_seqlenth])
    biaffine = tf.transpose(biaffine, (0, 1,3,2))
    if bias[2]:
        bias_w = tf.get_variable(
               shape=[out_size],
                name = "bias_bi" + name,
                initializer= tf.zeros_initializer,
                dtype=tf.float32)
        biaffine = tf.map_fn(lambda x : tf.add(x, bias_w), tf.reshape(biaffine, [-1, out_size]))
        biaffine = tf.reshape(biaffine, [batch_size, -1, batch_seqlenth, out_size])
    biaffine = tf.reshape(biaffine, [-1, batch_seqlenth, out_size])

    return tf.gather(biaffine, verbs_for_index)


def Biaffine(input1, input2, out_size ,name,dim,  FLAGS,batch_seqlenth,
                 bias=(True, True, True)):
    dim1 = dim2 = dim
    if bias[0]:
            ones1 = tf.fill([tf.shape(input1)[0], tf.shape(input1)[1], 1], 1.0)   ##这里fill支持动态len1
            input1 = tf.concat([input1, ones1], axis=2)
            dim1 += 1
    if bias[1]:
            ones2 = tf.fill([tf.shape(input2)[0], tf.shape(input2)[1], 1], 1.0)
            input2 = tf.concat([input2, ones2], axis=2)
            dim2 += 1

    W = tf.get_variable(
                "W_bi_" + name,
                shape=[dim1, dim2 * out_size],
                dtype=tf.float32,
                initializer= tf.truncated_normal_initializer)
    input1_reshaped = tf.reshape(input1, [-1, dim1])
    affine = tf.matmul(input1_reshaped, W)
    affine = tf.reshape(affine, ([FLAGS.batch_size, -1, dim2]))
    biaffine   = tf.matmul(affine, tf.transpose(input2, (0,2,1)))
    biaffine = tf.reshape(biaffine, [FLAGS.batch_size, -1, out_size, batch_seqlenth])
    biaffine = tf.transpose(biaffine, (0, 1,3,2))
    if bias[2]:
        bias_w = tf.get_variable(
               shape=[out_size],
                name = "bias_bi" + name,
                initializer= tf.zeros_initializer,
                dtype=tf.float32)
        biaffine = tf.map_fn(lambda x : tf.add(x, bias_w), tf.reshape(biaffine, [-1, out_size]))
        biaffine = tf.reshape(biaffine, [FLAGS.batch_size, -1, batch_seqlenth, out_size])
    return biaffine


class DIST_FN:
    '''  function for the same operation on a list of tensors with not the same length'''

    def __init__(self, fns, names, args=None):
        self.first_use = 1
        assert isinstance(fns, list), "'fns' must be a list of function."
        self.fns = fns
        self.names = names
        assert len(self.fns) == len(self.names), "'fns' lenght must be with names."
        self.args = args
        self.args_length = 0
        if  self.args:
            self.args_length = len(self.args)
            for item in self.args:
                assert isinstance(self.args[0], list), "'args'inside part must be a list of params."

    def use_fns(self, inputs):
            arg = list()
            results = inputs
            if self.first_use  == 1:
                self.first_use = 0
                for i in range(len(self.fns)):
                    if i+1 <= self.args_length:
                        arg = self.args[i]
                    results = self.fns[i](results, *arg, name=self.names[i])
                return results
            else:
                for i in range(len(self.fns)):
                    if self.args_length and i+1 <= self.args_length:
                        arg = self.args[i]
                    try:
                        results = self.fns[i](results, *arg, reuse=True,name=self.names[i])
                    except:
                        results = self.fns[i](results, *arg, name=self.names[i])
                return results


def time_distributed(fns, incoming, names, time_major = False, args=None):
    '''
    incoming = [tf.constant([[[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]]]),tf.constant([[[0.3,0.4],[0.5,0.6],[0.7,0.8]]])]
    :param fn:
    :param incoming:
    :param name:
    :param args:
    :return:
    '''
    if not time_major:
        incoming = tf.transpose(incoming, [1, 0, 2])
    fn_dis = DIST_FN(fns, names, args)
    #results = [fn_dis.use_fns(i) for i in incoming]
    results = tf.map_fn(fn_dis.use_fns, incoming)

    return tf.transpose(results, [1, 0, 2])
