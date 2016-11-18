import lstm
import numpy as np
import random

__author__ = 'vassilis'

SEQUENCE_FUNCTIONS = ["none", "none", "none"]

class Multi_Layer_LSTM:
    def __init__(self,max_depth, hidden_layer_sizes, input_sizes):
        self.lstm_stack = [lstm.LSTM() for l in range(max_depth)]
        for l in range(max_depth):
            self.lstm_stack[l].initialize(input_sizes[l] + (0 if l== max_depth -1 else hidden_layer_sizes[l + 1]), hidden_layer_sizes[l])
        self.hidden_layer_sizes = hidden_layer_sizes
        self.input_sizes = input_sizes


    """ Forward instance function through the multi layer LSTM architecture"""
    def forward_instance(self, instance_node, current_depth, max_depth, sequence_function = SEQUENCE_FUNCTIONS):
        if instance_node.get_number_of_children() == 0:
            return -100 * np.ones(self.hidden_layer_sizes[current_depth]) # no children signifier vector
        input_sequence = np.array([])
        children_sequence = get_sequence(instance_node.get_children(), sequence_function[current_depth])
        for item in children_sequence:
            feature_vector = item.get_feature_vector()
            """ If we are not at the very bottom we need to get input from LSTM at the next level"""
            LSTM_output_from_below = np.array([])
            if current_depth < max_depth:
                 LSTM_output_from_below = self.forward_instance(item, current_depth + 1, max_depth).reshape(self.hidden_layer_sizes[current_depth +1]) # recursive call
            full_feature_vector = np.concatenate((LSTM_output_from_below, feature_vector)) # concatenate feature vector and input from LSTM output below
            # concatenate current feature vector to input sequence for the LSTM
            input_sequence = np.concatenate((input_sequence,full_feature_vector))
        # forward the input sequence to this depth's LSTM
        input_sequence = input_sequence.reshape(instance_node.get_number_of_children(), 1, len(full_feature_vector))
        _, _, Y, cache = self.lstm_stack[current_depth]._forward(input_sequence)
        instance_node.cache = cache
        instance_node.children_sequence = children_sequence
        return softmax(Y)


    def calculate_backward_gradients(self,instance_node, derivative, current_depth, max_depth):
        dX, g, _, _ = self.lstm_stack[current_depth].backward_return_vector_no_update(d = derivative, cache = instance_node.cache)
        instance_node.gradient = g
        if current_depth == max_depth:
            return
        counter = 0
        for item in instance_node.children_sequence:
            if item.cache is None:
                continue
            self.calculate_backward_gradients(item, dX[counter,:,0:self.hidden_layer_sizes[current_depth + 1]], current_depth + 1, max_depth = max_depth)
            counter += 1


    def update_LSTM_weights(self,instance_node, current_depth, max_depth, learning_rate_vector):
        if not instance_node.gradient is None:
            self.lstm_stack[current_depth].WLSTM -= learning_rate_vector[current_depth] * instance_node.gradient
        if current_depth == max_depth:
            return
        for item in instance_node.children_sequence:
            self.update_LSTM_weights(item, current_depth + 1, max_depth, learning_rate_vector)


    def sgd_train_multilayer(self, root, target, max_depth, objective_function, learning_rate_vector):
        # first pass the instance root one forward so that all internal LSTM states
        # get calculated and stored in "cache" field
        Y = self.forward_instance(root, current_depth = 0, max_depth= max_depth)
        deriv = getDerivative(output = Y, target = target, objective = objective_function)
        self.calculate_backward_gradients(root, deriv, 0, max_depth)
        self.update_LSTM_weights(root, 0, max_depth, learning_rate_vector = learning_rate_vector)

    def train_model_force_balance(self, training_set, no_of_instances, max_depth, objective_function, learning_rate_vector):
        counter = 0
        if no_of_instances == 0:
            return
        for item in get_balanced_training_set(training_set, self.hidden_layer_sizes[0]):
            if item.get_number_of_children() == 0:
                continue
            target = np.zeros((1,self.hidden_layer_sizes[0]))
            target[0,item.get_label()] = 1.0
            self.sgd_train_multilayer(item, target, max_depth, objective_function, learning_rate_vector)
            counter += 1
            if counter % 10000 == 0:
                print "Training has gone over", counter, " instances.."
            if counter == no_of_instances:
                break

    def test_model_simple(self, test_set, max_depth):
        guesses = 0
        hits = 0
        found = {}
        missed = {}
        misclassified = {}
        for item in test_set:
            Y = self.forward_instance(item, 0 , max_depth)
            if Y is None:
                continue
            #print Y
            predicted_label = Y.argmax()
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
        print "LSTM results"
        print "============================================================="
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
            if (no_of_finds + no_of_misclassified) == 0:
                precision = 0
            else:
                precision = no_of_finds / (no_of_finds + no_of_misclassified)
            recall_dict[label] = recall
            precision_dict[label] = precision
            recall_list.append(recall)
        print "Average recall ", np.mean(recall_list)
        if len(all_labels) == 2: # compute F-1 score for binary classification
            for label in all_labels:
                print "Precision for label ", label, " is :", precision_dict[label]
                print "Recall for label ", label, " is :", recall_dict[label]
                print "F-1 score for label ", label, " is : ", 0 if (precision_dict[label] + recall_dict[label]) == 0 else 2* (precision_dict[label] * recall_dict[label])/ (precision_dict[label] + recall_dict[label])




class Instance_node:
    def __init__(self, feature_vector = None, label = None, id = None, txout = None):
        self.id = id
        self.txout = txout
        self.feature_vector = feature_vector
        self.label = label # an integer that represents the category of the item
        self.cache = None
        self.gradient = None
        self.children = []
        self.children_sequence = [] # Stores the specific order by which the items were fed into the LSTM to update weights correctly

    def get_number_of_children(self):
        return len(self.children)

    def get_label(self):
        return self.label

    def get_children(self):
        return self.children

    def get_feature_vector(self):
        return self.feature_vector


""" Generator that returns items from training set
    equally balanced among classes"""
def get_balanced_training_set(training_set, no_of_classes):
    # make bucket of classes to sample from
    buckets = {}
    buckets_current_indexes ={}
    for i in range(0, no_of_classes):
        buckets[i] = []
        buckets_current_indexes[i] = 0
    for item in training_set:
        category = item.get_label()
        buckets[category].append(item)
    while True:
        for i in range(0,no_of_classes):
            if len(buckets[i]) == 0: # if a class has no representatives, continue
                continue
            if buckets_current_indexes[i] == len(buckets[i]):
                buckets_current_indexes[i] = 0
            yield buckets[i][buckets_current_indexes[i]]
            buckets_current_indexes[i] += 1

def get_sequence(children_list, sequence_function):
    if sequence_function == "shuffle":
        res = list(children_list)
        random.shuffle(res)
    if sequence_function == "none":
        res = list(children_list)
    return res


def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

def getDerivative(output, target, objective):
    if objective == "softmax_classification":
        return output - target

