#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:zhanggguoxiao
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from bert import modeling
from bert import optimization
from bert import tokenization
from tensorflow.contrib.layers.python.layers import initializers
from base_model import *
import tensorflow as tf
from sklearn.metrics import f1_score,precision_score,recall_score
from tensorflow.python.ops import math_ops
import tf_metrics
import pickle
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", None, "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False,"Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float('droupout_rate', 0.1, 'Dropout rate')
flags.DEFINE_bool("add_lstm", False, "Whether to add bi-lstm")


flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        #self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            for line in f:
                words = line.strip().split('\t')[1]
                labels = line.strip().split('\t')[0]
                lines.append((labels, words))
            return lines


class SrlProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")


    def get_labels(self):
        return [ 'O', "X","[CLS]","[SEP]", 'B_C-ARGM-DSP', 'B_C-ARGM-COM', 'B_R-ARG3', 'I_R-ARGM-GOL', 'I_R-ARG3', 'I_ARGM-PRP', 'B_ARGM-PRD', 'B_C-ARGM-CAU', 'B_ARGM-DIR', 'I_ARG2', 'B_C-ARG2', 'B_R-ARGM-PRP', 'B_ARGM-PRP', 'B_ARG1', 'I_C-ARG4', 'B_R-ARGM-TMP', 'I_ARGM-DIR', 'B_C-ARGM-TMP', 'B_C-ARGM-DIS', 'I_ARGM-LOC', 'B_R-ARGM-DIR', 'B_ARGM-LVB', 'I_R-ARGM-PRP', 'B_C-ARGM-ADV', 'I_ARGM-COM', 'I_ARG3', 'B_ARGM-LOC', 'B_ARGM-MNR', 'B_R-ARGM-CAU', 'B_C-ARGM-NEG', 'I_C-ARGM-DIR', 'I_C-ARGM-EXT', 'B_C-ARGM-PRP', 'I_R-ARGM-LOC', 'B_ARG4', 'I_C-ARG0', 'I_R-ARGM-PRD', 'I_C-ARGM-TMP', 'I_C-ARG2', 'B_C-ARGM-MOD', 'I_R-ARG0', 'B_ARGM-CAU', 'I_C-ARGM-COM', 'I_ARGM-ADJ', 'V', 'I_R-ARGM-TMP', 'B_R-ARG5', 'B_C-ARGM-ADJ', 'B_R-ARGM-LOC', 'I_R-ARGM-PNC', 'B_R-ARGM-PNC', 'I_C-ARGM-DSP', 'B_C-ARG3', 'I_C-ARGM-LOC', 'I_C-ARGM-CAU', 'B_ARGM-COM', 'I_ARGM-DIS', 'B_ARGM-TMP', 'I_ARGM-ADV', 'B_R-ARG2', 'B_ARGM-PRR', 'B_ARGM-NEG', 'I_C-ARGM-DIS', 'B_ARGM-MOD', 'B_ARG5', 'I_C-ARGM-MOD', 'B_ARGM-PRX', 'I_ARG4', 'B_R-ARGM-COM', 'I_C-ARGM-NEG', 'I_C-ARG1', 'B_ARGA', 'I_ARGM-REC', 'I_ARGM-MOD', 'I_ARG1', 'B_ARG3', 'I_ARGM-GOL', 'B_C-ARGM-LOC', 'B_C-ARG1', 'I_R-ARGM-CAU', 'B_ARG2', 'I_R-ARGM-EXT', 'B_C-ARGM-MNR', 'B_R-ARGM-ADV', 'I_R-ARG1', 'I_R-ARGM-COM', 'B_R-ARG0', 'I_C-ARGM-PRP', 'B_ARGM-EXT', 'I_R-ARG2', 'B_ARG0', 'I_ARGM-PNC', 'I_R-ARGM-ADV', 'I_ARGM-DSP', 'B_ARGM-DIS', 'I_ARG0', 'I_R-ARG4', 'B_R-ARGM-MNR', 'I_ARGM-NEG', 'B_ARGM-PNC', 'I_ARGM-EXT', 'B_R-ARGM-PRD', 'I_R-ARGM-MNR', 'I_C-ARG3', 'B_R-ARGM-EXT', 'B_ARGM-DSP', 'B_C-ARG4', 'I_ARG5', 'I_ARGM-PRD', 'B_C-ARG0', 'I_C-ARGM-ADV', 'I_R-ARGM-DIR', 'B_R-ARGM-GOL', 'B_ARGM-ADV', 'I_C-ARGM-ADJ', 'B_R-ARG4', 'I_ARGA', 'I_ARGM-MNR', 'B_R-ARG1', 'B_ARGM-REC', 'I_ARGM-CAU', 'B_ARGM-GOL', 'B_ARGM-ADJ', 'B_C-ARGM-DIR', 'I_ARGM-TMP', 'I_C-ARGM-MNR', 'B_C-ARGM-EXT', 'B_R-ARGM-MOD']
        # return ['O', "X","[CLS]","[SEP]", 'E_R-ARG0', 'S_C-ARG0', 'I_R-ARG1', 'S_ARGM-MOD', 'E_ARGM-PRD', 'B_R-ARGM-CAU', 'E_ARGM-MOD',
        #         'I_C-ARGM-DIR', 'S_R-ARG4', 'E_ARG0', 'B_ARGM-ADJ', 'I_C-ARG4', 'B_ARG2', 'S_ARGM-PRP', 'I_ARGM-DSP', 'B_R-ARGM-LOC',
        #         'E_C-ARGM-DSP', 'S_R-ARG5', 'B_R-ARGM-GOL', 'E_R-ARG2', 'I_ARGA', 'I_C-ARGM-CAU', 'B_ARGM-GOL', 'B_C-ARGM-PRP', 'I_ARGM-PRD',
        #         'E_ARGM-DIS', 'S_R-ARGM-DIR', 'E_C-ARGM-LOC', 'B_R-ARGM-PNC', 'S_C-ARGM-MNR', 'I_R-ARG0', 'I_R-ARGM-PNC', 'S_R-ARG0', 'B_C-ARGM-EXT',
        #         'E_C-ARGM-EXT', 'S_R-ARGM-EXT', 'E_C-ARGM-DIR', 'E_ARGM-REC', 'E_ARGM-NEG', 'B_R-ARGM-DIR', 'E_ARGM-ADJ', 'B_ARGM-REC', 'S_ARGM-PRR', 'B_C-ARGM-COM', 'B_C-ARGM-ADV', 'E_ARGM-COM', 'I_ARGM-TMP', 'S_R-ARGM-LOC', 'B_ARGM-ADV', 'I_ARGM-COM', 'B_R-ARG2', 'S_ARGA', 'I_ARGM-EXT', 'B_R-ARG0', 'B_ARGM-MOD', 'B_ARGA', 'B_ARGM-PNC', 'E_ARGM-DIR', 'I_ARG1', 'S_ARGM-EXT', 'I_ARGM-ADJ', 'E_C-ARGM-ADJ', 'B_C-ARG3', 'I_ARGM-MOD', 'E_ARGM-GOL', 'I_C-ARG2', 'S_ARGM-PRD', 'I_ARGM-DIR', 'I_ARGM-PNC', 'B_R-ARGM-PRP', 'E_C-ARGM-ADV', 'S_ARGM-DSP', 'B_C-ARGM-CAU', 'S_ARGM-PRX', 'I_C-ARGM-NEG', 'S_C-ARG2', 'E_R-ARG3', 'S_ARGM-DIR', 'E_R-ARGM-PRP', 'I_C-ARGM-DIS', 'E_ARGM-MNR', 'S_R-ARGM-MNR', 'I_ARGM-GOL', 'I_C-ARGM-MNR', 'B_C-ARGM-MOD', 'S_ARGM-REC', 'E_C-ARGM-NEG', 'E_ARG5', 'S_ARGM-DIS', 'S_ARGM-GOL', 'B_C-ARG0', 'S_ARGM-CAU', 'B_R-ARG1', 'I_C-ARGM-DSP', 'B_ARGM-PRP', 'E_R-ARG4', 'I_C-ARGM-TMP', 'S_R-ARGM-CAU', 'B_C-ARGM-DIS', 'B_C-ARGM-LOC', 'V', 'B_R-ARGM-PRD', 'I_ARGM-ADV', 'B_R-ARGM-COM', 'E_R-ARG1', 'S_ARG2', 'B_C-ARGM-MNR', 'B_C-ARGM-TMP', 'B_C-ARGM-DIR', 'S_ARG1', 'E_ARGM-PNC', 'E_R-ARGM-PRD', 'E_R-ARGM-PNC', 'B_C-ARG1', 'I_ARGM-PRP', 'E_ARGM-DSP', 'S_ARGM-LOC', 'S_ARG0', 'E_R-ARGM-LOC', 'E_ARGM-ADV', 'S_ARGM-NEG', 'B_ARGM-TMP', 'B_ARG1', 'E_R-ARGM-GOL', 'I_C-ARGM-ADV', 'I_C-ARGM-LOC', 'E_C-ARG4', 'E_ARGM-PRP', 'S_R-ARGM-PNC', 'S_ARGM-ADV', 'E_C-ARGM-TMP', 'S_R-ARGM-TMP', 'E_ARGM-LOC', 'I_C-ARG3', 'B_C-ARG4', 'E_C-ARGM-MNR', 'B_C-ARG2', 'E_ARGM-TMP', 'S_ARG4', 'B_ARGM-COM', 'E_ARGM-EXT', 'S_ARGM-TMP', 'B_ARGM-EXT', 'B_ARGM-DSP', 'E_C-ARGM-PRP', 'I_ARGM-REC', 'B_ARG4', 'I_C-ARG1', 'S_ARG5', 'B_ARGM-NEG', 'S_R-ARGM-PRP', 'S_R-ARGM-MOD', 'S_ARG3', 'E_R-ARGM-COM', 'I_R-ARG2', 'E_ARG1', 'I_ARG0', 'S_R-ARG1', 'I_R-ARG3', 'S_ARGM-MNR', 'E_ARG4', 'E_C-ARG0', 'B_R-ARGM-TMP', 'E_C-ARGM-MOD', 'B_R-ARGM-EXT', 'E_ARGM-CAU', 'B_ARGM-DIS', 'I_ARGM-LOC', 'I_R-ARGM-LOC', 'E_R-ARGM-EXT', 'I_ARG2', 'S_R-ARG2', 'E_C-ARGM-DIS', 'I_C-ARGM-PRP', 'S_ARGM-COM', 'B_C-ARGM-DSP', 'I_ARG3', 'E_C-ARG2', 'B_C-ARGM-ADJ', 'I_ARGM-CAU', 'I_C-ARG0', 'B_ARG5', 'E_R-ARGM-MNR', 'E_ARGA', 'S_ARGM-PNC', 'I_ARG5', 'B_R-ARG4', 'E_C-ARG1', 'S_C-ARGM-TMP', 'E_ARG3', 'E_R-ARGM-CAU', 'E_C-ARGM-CAU', 'E_R-ARGM-TMP', 'S_ARGM-ADJ', 'B_ARGM-MNR', 'I_ARGM-NEG', 'I_C-ARGM-COM', 'B_ARGM-DIR', 'B_R-ARGM-MNR', 'B_C-ARGM-NEG', 'E_C-ARG3', 'I_ARGM-DIS', 'I_C-ARGM-EXT', 'B_ARGM-CAU', 'E_C-ARGM-COM', 'B_ARG3', 'S_R-ARGM-ADV', 'B_ARGM-PRD', 'S_C-ARG1', 'I_ARGM-MNR', 'B_R-ARGM-ADV', 'E_R-ARGM-ADV', 'B_ARGM-LOC', 'S_ARGM-LVB', 'S_R-ARGM-COM', 'B_ARG0', 'E_ARG2', 'E_R-ARGM-DIR', 'I_ARG4', 'S_R-ARG3', 'B_R-ARG3']


    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def write_tokens(tokens,mode):
    if mode=="test":
        path = os.path.join(FLAGS.output_dir, "token_"+mode+".txt")
        wf = open(path,'a')
        for token in tokens:
            if token!="**NULL**":
                wf.write(token+'\n')
        wf.close()

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,mode):
    label_map = {}
    for (i, label) in enumerate(label_list,1):
        label_map[label] = i
    with open(os.path.join(FLAGS.output_dir,'label2id.pkl'),'wb') as w:
        pickle.dump(label_map,w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    #label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        #label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    #assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        #tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        #label_mask = label_mask
    )
    write_tokens(ntokens,mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file,mode=None
):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        #features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        #"label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn

def length(data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
        
def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    sent_features = model.get_sequence_output()
    seq_lenths = length(input_ids)
    with tf.variable_scope("lstm-crf"):
        if is_training:
            sent_features = tf.nn.dropout(sent_features, keep_prob=1-FLAGS.droupout_rate)
        if FLAGS.add_lstm:
            hidden_size = sent_features.shape[-1].value
            sent_features = BiLSTM(sent_features, 1-FLAGS.droupout_rate, seq_lenths, hidden_size)
        def fn(inputs, name=None, reuse = False):
            return tf.layers.dense(inputs, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                            units = num_labels, name=name, reuse= reuse)
        unary_scores = time_distributed([fn],sent_features,['dense'], [])
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[num_labels, num_labels],
                initializer=initializers.xavier_initializer())
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                inputs=unary_scores,
                tag_indices=labels,
                transition_params=trans,
                sequence_lengths=seq_lenths)
    total_loss = tf.reduce_mean(-log_likelihood)
    pred_ids, _ = tf.contrib.crf.crf_decode(potentials=unary_scores,
                                            transition_params=transition_params, sequence_length=seq_lenths)
    return (total_loss, transition_params, unary_scores, pred_ids)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        #label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, transition_params, unary_scores,pred_ids) = create_model(
            bert_config, is_training, input_ids, input_mask,segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            
            def metric_fn(label_ids, unary_scores,  transition_params ):
            # def metric_fn(label_ids, logits):
                included_label_ids = list(range(1,num_labels)[5:])
                weight = tf.sequence_mask(FLAGS.max_seq_length)
                precision = tf_metrics.precision(label_ids,pred_ids,num_labels,included_label_ids, weight)
                recall = tf_metrics.recall(label_ids,pred_ids,num_labels,included_label_ids, weight)
                f = tf_metrics.f1(label_ids,pred_ids,num_labels,included_label_ids, weight)
                #
                return {
                    "eval_precision":precision,
                    "eval_recall":recall,
                    "eval_f": f,
                    #"eval_loss": loss,
                }
            eval_metrics = (metric_fn, [label_ids, unary_scores, transition_params])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode,predictions= pred_ids,scaffold_fn=scaffold_fn
            )
        return output_spec
    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "srl": SrlProcessor,
    }
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list)+1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        with open(os.path.join(FLAGS.output_dir,'label2id.pkl'),'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value:key for key,value in label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file,mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        with open(output_predict_file,'w') as writer:
            for prediction in result:
                output_line = "\n".join(id2label[id] for id in prediction if id!=0) + "\n"
                writer.write(output_line)

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()


