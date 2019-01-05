# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import array_ops



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
    results = tf.map_fn(fn_dis.use_fns, incoming)

    return tf.transpose(results, [1, 0, 2])
