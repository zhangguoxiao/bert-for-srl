import numpy as np
import codecs
import pickle
import os
import argparse
from bert import tokenization


def init_map_join_srl(labels):
    items = labels
    new_items = set()
    for item in items:
        if item not in  ["V",'O', "X","[CLS]","[SEP]"]:
            new_item = item.split("_")[1]
            new_items.add(new_item)
    results = {}
    for item in list(new_items):
        results[item] = {}
        results[item]["total_count"] = 0
        results[item]["pre_right_count"] = 0
        results[item]["pre_total_count"] = 0
    results["all"] = {}
    results["all"]["total_count"] = 0
    results["all"]["pre_right_count"] = 0
    results["all"]["pre_total_count"] = 0
    return results


def add_map_join_complex(results, sent_pre_tags, sent_true_tags, use_post_process = True):
    if  len(sent_true_tags) != len(sent_pre_tags):
        print("error compared items: %s \n %s" % (" ".join(sent_true_tags), " ".join(sent_pre_tags)))

    for j in range(len(sent_pre_tags)):
        if sent_pre_tags[j].startswith("S_") or sent_pre_tags[j].startswith("B_"):
            new_item = sent_pre_tags[j].split("_")[1]
            results[new_item]["pre_total_count"] += 1
            results["all"]["pre_total_count"] += 1

    for i in range(len(sent_true_tags)):
        item = sent_true_tags[i]
        if item.startswith("S_"):
            new_item = item.split("_")[1]
            results[new_item]["total_count"] += 1
            results["all"]["total_count"] += 1
            if sent_pre_tags[i] == item:
                        results[new_item]["pre_right_count"] += 1
                        results["all"]["pre_right_count"] += 1
            if use_post_process and sent_pre_tags[i]== "B_" + new_item:
                k = i + 1
                while(k < len(sent_pre_tags) and sent_pre_tags[k] == "X"):
                    k += 1
                if k == len(sent_pre_tags):
                    results[new_item]["pre_right_count"] += 1
                    results["all"]["pre_right_count"] += 1
                if k < len(sent_pre_tags) and not sent_pre_tags[k].endswith(new_item):
                    results[new_item]["pre_right_count"] += 1
                    results["all"]["pre_right_count"] += 1

        if item.startswith("B_"):
            new_item = item.split("_")[1]
            results[new_item]["total_count"] += 1
            results["all"]["total_count"] += 1
            if sent_pre_tags[i] == item:
                    k = i + 1
                    wrong = 0
                    while k < len(sent_true_tags) and (not sent_true_tags[k].startswith("E_")):
                        if ("_" not in sent_pre_tags[k] and sent_pre_tags[k] != "X"):
                            wrong = 1
                        if "_"  in sent_pre_tags[k] and sent_pre_tags[k] != "I_" + new_item:
                            wrong = 1
                        k += 1
                    if k == len(sent_true_tags):
                        pass
                    else:
                        if sent_pre_tags[k] != "E_" + new_item:
                            if sent_pre_tags[k].endswith(new_item):
                                k1 = k + 1
                                while(k1 < len(sent_pre_tags) and sent_pre_tags[k1] == "X"):
                                    k1 += 1
                                if k1 < len(sent_pre_tags) and sent_pre_tags[k1].endswith(new_item):
                                    wrong = 1
                            else:
                                wrong = 1
                    if wrong == 0:
                            results[new_item]["pre_right_count"] += 1
                            results["all"]["pre_right_count"] += 1

    return results

def add_map_join_simple(results, sent_pre_tags, sent_true_tags):
    if  len(sent_true_tags) != len(sent_pre_tags):
        print("error compared items: %s \n %s" % (" ".join(sent_true_tags), " ".join(sent_pre_tags)))


    for idx in range(len(sent_pre_tags)):
        sent_pre_tags[idx] = sent_pre_tags[idx].replace("S_", "B_").replace("E_", "I_")
        sent_true_tags[idx] = sent_true_tags[idx].replace("S_", "B_").replace("E_", "I_")

    for j in range(len(sent_pre_tags)):
        if sent_pre_tags[j].startswith("B_"):
            new_item = sent_pre_tags[j].split("_")[1]
            results[new_item]["pre_total_count"] += 1
            results["all"]["pre_total_count"] += 1

    for i in range(len(sent_true_tags)):
        item = sent_true_tags[i]
        if item.startswith("B_"):
            new_item = item.split("_")[1]
            results[new_item]["total_count"] += 1
            results["all"]["total_count"] += 1
            if sent_pre_tags[i] == item:
                    k = i + 1
                    wrong = 0
                    while k < len(sent_true_tags) and (sent_true_tags[k] in ["I_" + new_item, "X"]):
                        if ("_" not in sent_pre_tags[k] and sent_pre_tags[k] != "X"):
                            wrong = 1
                        if "_"  in sent_pre_tags[k] and sent_pre_tags[k] != "I_" + new_item:
                            wrong = 1
                        k += 1
                    if k == len(sent_true_tags):
                        pass
                    else:
                        if sent_pre_tags[k] in ["I_" + new_item, "X"]:
                                wrong = 1
                    if wrong == 0:
                            results[new_item]["pre_right_count"] += 1
                            results["all"]["pre_right_count"] += 1

    return results


def add_map_join(results, sent_pre_tags, sent_true_tags, data_type):
    if data_type=='simple':
        return add_map_join_simple(results, sent_pre_tags, sent_true_tags)
    else:
        return add_map_join_complex(results, sent_pre_tags, sent_true_tags)


def gen_final_reports(results):
    final_value = 0.0
    print("\n******************************************")
    for key in results:
             if results[key]["pre_total_count"] == 0:
                 presition = 0
             else:
                 presition = results[key]["pre_right_count"]/float(results[key]["pre_total_count"])
             if results[key]["total_count"] == 0:
                 recall = 0
             else:
                recall = results[key]["pre_right_count"]/float(results[key]["total_count"])
             if (presition + recall) == 0:
                 f_value = 0
             else:
                f_value = float(2 * presition *recall)/(presition + recall)
             if key == "all":
                 final_value = f_value
             print(key + "  presition: %.5f  recall: %.5f   fvalue: %.5f" %(presition, recall, f_value))
    print("******************************************\n")
    return  final_value


def change(tokenizer, textlist, labellist, max_seq_length):
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
    if len(tokens) >= max_seq_length - 1:
        labels = labels[0:(max_seq_length - 2)]
    return labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
          '--output_dir',
          type=str,
          default= "/home/lion/experiments/zgx/bert_srl/output/",
          help='output_dir'
    )

    parser.add_argument(
          '--data_dir',
          type=str,
          default= "/home/lion/experiments/zgx/bert_srl/SRLdata",
          help='data_dir'
    )
    parser.add_argument(
          '--data_type',
          type=str,
          default= "simple",
          help='bio:simple;bieso:complex'
    )
    parser.add_argument(
          '--max_seq_length',
          type=int,
          default= 128,
          help='max_seq_length'
    )
    parser.add_argument(
          '--do_lower_case',
          type=bool,
          default= True,
          help='Whether to lower case the input text.'
    )
    parser.add_argument(
          '--vocab_loc',
          type=str,
          default= "checkpoint/",
          help='The vocabulary file that the BERT model was trained on.'
    )
    FLAGS, _ = parser.parse_known_args()
    output_dir = FLAGS.output_dir
    data_dir = FLAGS.data_dir
    max_seq_length = FLAGS.max_seq_length
    do_lower_case = FLAGS.do_lower_case
    vocab_file = os.path.join(FLAGS.vocab_loc, 'vocab.txt')
    data_type = FLAGS.data_type
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=FLAGS.do_lower_case)

    with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'rb') as rf:
                label2id = pickle.load(rf)
    labels = list(label2id.keys())
    results = init_map_join_srl(labels)
    pre_labes_total = []
    pre_labels = []
    with codecs.open(os.path.join(output_dir , 'label_test.txt'),encoding='utf-8') as f:
        for line in f:
            if line.strip() != "[SEP]":
                pre_labels.append(line.strip())
            else:
                pre_labes_total.append(pre_labels[1:])
                pre_labels = []
    true_labels_total = []
    with codecs.open(os.path.join(data_dir, 'test.tsv'),encoding='utf-8') as g:
        for line in g:
            if len(line.strip()) > 0:
                textlist = line.strip().split("\t")[1].split(" ")
                labellist = line.strip().split("\t")[0].split(" ")
                true_labels = change(tokenizer, textlist, labellist, max_seq_length)
                true_labels_total.append(true_labels)

    for i in range(len(true_labels_total)):
        add_map_join(results, pre_labes_total[i], true_labels_total[i], data_type)

    gen_final_reports(results)
