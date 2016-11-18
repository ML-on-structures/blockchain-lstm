__author__ = 'vassilis'

import multi_layer_LSTM as ml
import numpy as np
import random
import math


distribution_of_children = [{} for _ in range(10)]

def getLabelsForAddresses(filename):
    labels_dict = {}
    with open(filename,'r') as csv:
        counter = 0
        for line in csv:
            if counter == 0: # First line contains attribute names
                attributes = line.strip().split(',')
                addr_index = attributes.index("addr")
                balance1_index = attributes.index("balance_at_timestamp1")
                time1_index = attributes.index("timestamp1")
                balance2_index = attributes.index("balance_at_timestamp2")
                time2_index = attributes.index("timestamp2")
                counter += 1
                continue
            content = line.strip().split(',')
            addr = int(content[addr_index])
            label = 0 if int(content[balance2_index]) == 0 else 1
            counter += 1
            labels_dict[addr] = label
    return labels_dict

def getTxFeatures(filename):
    tx_dict = {}
    with open(filename,'r') as csv:
        counter = 0
        for line in csv:
            if counter == 0: # First line contains attribute names
                attributes = line.strip().split(',')
                address_index = attributes.index("address")
                interacting_address_index = attributes.index("interacting_address")
                txout_index = attributes.index("txout")
                incoming_index = attributes.index("incoming")
                lock_time_index = attributes.index("lock_time")
                coinbase_index = attributes.index("coinbase")
                tx_size_index = attributes.index("tx_size")
                satoshi_value_index = attributes.index("satoshi_value")
                total_satoshi_index = attributes.index("total_satoshi")
                fraction_index = attributes.index("value_fraction")
                total_txout_count_index = attributes.index("total_txout_count")
                spent_already_index = attributes.index("spent_already")
                counter += 1
                continue
            content = line.strip().split(',')
            addr = int(content[address_index])
            interacting_addr = int(content[interacting_address_index])
            # construct feature vector
            feature_vector = np.zeros(9)
            feature_vector[0] = 1.0 if content[incoming_index]=='t' else 0.0
            feature_vector[1] = (float(content[lock_time_index]) - 1230841628) / 200000000.0
            feature_vector[2] = 1.0 if content[coinbase_index]=='t' else 0.0
            feature_vector[3] = 1 - math.exp(-int(content[tx_size_index])/ 1000)
            feature_vector[4] = 1 - math.exp(-int(content[satoshi_value_index])/ 100000)
            feature_vector[5] = 1 - math.exp(-int(content[total_satoshi_index])/ 100000)
            feature_vector[6] = float(content[fraction_index])
            feature_vector[7] = int(content[total_txout_count_index]) / 100
            feature_vector[8] = 1.0 if content[spent_already_index] == 't' else 0.0
            if addr not in tx_dict:
                tx_dict[addr] = [(interacting_addr, feature_vector)]
            else:
                tx_dict[addr].append((interacting_addr, feature_vector))
            counter += 1
    return tx_dict


def build_subtree(instance_node, txs, current_depth, max_depth):
    count_kids = 0
    if instance_node.id in txs[current_depth]:
        for inter_addr, tx in txs[current_depth][instance_node.id]:
            new_child = ml.Instance_node(id = inter_addr)
            new_child.feature_vector = tx.copy()
            instance_node.children.append(new_child)
            count_kids += 1
            if current_depth < max_depth - 1:
                build_subtree(new_child, txs, current_depth + 1, max_depth)
    if count_kids not in distribution_of_children[current_depth]:
        distribution_of_children[current_depth][count_kids] = 1
    else:
        distribution_of_children[current_depth][count_kids] += 1


def get_data_multi_layers(filename_list, max_depth):
    labels = getLabelsForAddresses(filename_list[0])
    txs = [{} for _ in range(max_depth)]
    for i in range(max_depth):
        txs[i] = getTxFeatures(filename_list[i+1])
    instance_list = []
    counter = 0
    for addr in labels:
        new_node = ml.Instance_node(label = labels[addr], id = addr)
        instance_list.append(new_node)
        build_subtree(instance_node= new_node, current_depth = 0, txs= txs, max_depth = max_depth)
        counter += 1
        if counter % 1000 == 0:
            print "Built ", counter, " instances so far.."
    return instance_list


def test_baseline(test_set, target_proportion):
    guesses = 0
    hits = 0
    found = {}
    missed = {}
    misclassified = {}
    for item in test_set:
        predicted_label = 1.0 if random.random()<target_proportion else 0.0
        real_label = item.get_label()
        #print "Predicted label ", predicted_label , " real label", real_label
        guesses += 1
        hits += 1 if predicted_label == real_label else 0
        if predicted_label == real_label:
            if real_label not in found:
                found[real_label] = 1
            else:
                found[real_label] += 1
        if predicted_label != real_label:
            if real_label not in missed:
                missed[real_label] = 1
            else:
                missed[real_label] += 1
            if predicted_label not in misclassified:
                misclassified[predicted_label] = 1
            else:
                misclassified[predicted_label] += 1
    print "Predicted correctly ", hits , "over ", guesses, " instances."
    recall_list = []
    recall_dict = {}
    precision_dict = {}
    found_labels = set(found.keys())
    missed_labels = set(missed.keys())
    all_labels = found_labels.union(missed_labels)
    for label in all_labels:
        no_of_finds = float((0 if label not in found else found[label]))
        no_of_missed = float((0 if label not in missed else missed[label]))
        no_of_misclassified = float((0 if label not in misclassified else misclassified[label]))
        recall =  no_of_finds / (no_of_finds + no_of_missed)
        precision = no_of_finds / (no_of_finds + no_of_misclassified)
        recall_dict[label] = recall
        precision_dict[label] = precision
        recall_list.append(recall)
    print "Average recall ", np.mean(recall_list)
    if len(all_labels) == 2: # compute F-1 score for binary classification
        for label in all_labels:
            print "F-1 score for label ", label, " is : ", 2* (precision_dict[label] * recall_dict[label])/ (precision_dict[label] + recall_dict[label])


def multi_layer():
    DEPTH = 2
    # building the tree unfoldings
    # the csv files can be found at the project's dataset folder
    # in the blockchain_data.zip file
    # The file can be found here: https://drive.google.com/open?id=0ByhyeddNVklEQ2FwOHFLbmE4alU
    instance_list = get_data_multi_layers(['data/labels.csv',
                           'data/first_level_tx_features.csv',
                           'data/second_level_tx_features.csv',
                           'data/third_level_tx_features.csv']
                          , max_depth= DEPTH)

    HIDDEN_LAYER_SIZES = [2, 2, 2]
    INPUT_SIZES = [9,9,9]
    LEARNING_RATE_VECTOR = [0.001,0.1, 0.5]
    OBJECTIVE_FUNCTION = "softmax_classification"
    lstm_stack = ml.Multi_Layer_LSTM(DEPTH, HIDDEN_LAYER_SIZES, INPUT_SIZES)
    random.seed(500)
    random.shuffle(instance_list)
    training_set_size = 10000
    training_set = instance_list[0:training_set_size]
    # get labels proportion
    label_count = 0.0
    for i in training_set:
        if i.get_label() == 1.0:
            label_count += 1.0
    label_proportion = label_count / training_set_size
    print "Label proportion: ", label_proportion
    test_set = instance_list[training_set_size:len(instance_list)]
    lstm_stack.train_model_force_balance(training_set, no_of_instances = 50000, max_depth= DEPTH - 1, objective_function= OBJECTIVE_FUNCTION, learning_rate_vector= LEARNING_RATE_VECTOR)
    lstm_stack.test_model_simple(test_set, max_depth = DEPTH - 1)
    print "============================================================="
    print "Results from baseline (random assignment of labels based on class proportion in training set)"
    print "============================================================="
    test_baseline(test_set, label_proportion)


if __name__ == '__main__':
    multi_layer()
