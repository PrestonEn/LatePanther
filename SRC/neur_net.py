#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Author: Preston Engstrom, pe12nh, 5228549
Date: March 2, 2016

A simple implementation of a single hidden layer,
feed-forward neural network.

Backpropigation and Resilient Propigation are implemnted
as training functions.

PEP 8 is gr8
"""

import pprint
import math
import numpy as np
import random
import copy
import time
import csv

current_milli_time = lambda: int(round(time.time() * 1000))
print current_milli_time()
random.seed(current_milli_time())
pp = pprint.PrettyPrinter(indent=1)


def create_dataset(file, attrib_col_start, attrib_col_end, class_col, label_col, header=True):
    values =  np.loadtxt(file, delimiter=',',
                         usecols=range(attrib_col_start,attrib_col_end+1),
                         skiprows=1)

    labels =  np.loadtxt(file, delimiter=',', usecols=[class_col], skiprows=1, dtype=str)
    nums = []
    for string in labels:
        nums.append([int(n) for n in list(string)])

    return [list(a) for a in zip(values.tolist(), nums)]
iris_data = create_dataset("../data/iris/iris_post_norm_0to1.csv",0,3,4,5)
cancer_data = create_dataset("../data/wisc/cancer_post_norm_0to1.csv",0,29,30,31)


def activation(x):
    return 1.0 / (1.0 + math.exp(-x))

def error_function(y):
    return y * (1.0 - y)

def writefile(filename, header, data):
    writer=csv.writer(open(filename,'wb'))
    writer.writerow(header)

    for row in data:
        writer.writerow(row)


class Network(object):
    """ Neural network
    Arguments:
    inputs -- number of input neurons
    hidden -- number of hidden layer neurons
    outputs -- number of output classes
    w_min -- minimum random range for generating initial weights
    w_max -- maximum random range for generating initial weights
    act_der -- tuple of function pointers for passing activation and derivitive

    """
    def __init__(self, inputs, hidden, outputs, w_min, w_max, activ, err, bias=1):
        # if bias = 1, we have a bias on the input-hidden
        # and hidden-output matrix
        self.bias = bias

        # add bias to input and hidden layers
        self.num_inputs = inputs + self.bias
        self.num_hidden = hidden + self.bias
        self.num_output = outputs

        # initialize the layer value lists
        self.inputs = np.ones(self.num_inputs)
        self.hiddens = np.ones(self.num_hidden)
        self.outputs = np.ones(self.num_output)

        # initialize the weight matrices to 0
        self.in_weights = np.random.uniform(w_min, w_max,
                                               (self.num_inputs,
                                                self.num_hidden))

        self.hidden_weights = np.random.uniform(w_min, w_max,
                                                   (self.num_hidden,
                                                    self.num_output))

        # build weight update overlay matrices
        self.in_weight_overlay = np.zeros((self.num_inputs,
                                           self.num_hidden))

        self.hidden_weight_overlay = np.zeros((self.num_hidden,
                                               self.num_output))

        self.activation = activ
        self.error_function = err


    def run(self, inputs):
        """
        Arguments:
        inputs --
        """
        # set inputs
        for i in xrange(self.num_inputs - self.bias):
            self.inputs[i] = inputs[i]

        # pass inputs to hiddens
        for h in xrange(self.num_hidden - self.bias):
            total = 0.0
            for i in xrange(self.num_inputs):
                total += self.inputs[i] * self.in_weights[i][h]
            self.hiddens[h] = self.activation(total)

        # pass hiddens to outputs
        for o in xrange(self.num_output):
            total = 0.0
            for h in xrange(self.num_hidden):
                total += self.hiddens[h] * self.hidden_weights[h][o]
            self.outputs[o] = self.activation(total)

        # map the results to a more readable predicted class
        max_value = max(self.outputs)
        max_index = self.outputs.tolist().index(max_value)
        predicted_class = np.zeros(len(self.outputs))
        predicted_class[max_index] = 1.0

        # return outputs and readable class
        return [self.outputs[:], predicted_class]

    def backprop(self, targets, learning, momentum):
        """Backpropigation
        Arguments:
        targets -- array of target values of form [0,0,1]
        learn_rt -- 0.0 < value <= 1.0, learning rate
        momentum -- 0.0 < value <= 1.0
        """
        out_deltas = np.zeros(self.num_output)
        hidden_deltas = np.zeros(self.num_hidden)

        # calulate out deltas
        for o in xrange(self.num_output):
            out_deltas[o] = targets[o] - self.outputs[o]
            out_deltas[o] *= self.error_function(self.outputs[o])

        # calculate hidden deltas
        for h in xrange(self.num_hidden):
            error = 0.0
            for o in xrange(self.num_output):
                error += out_deltas[o] * self.hidden_weights[h][o]
            hidden_deltas[h] = error * self.error_function(self.hiddens[h])

        # update hidden weights
        for h in xrange(self.num_hidden):
            for o in xrange(self.num_output):
                change = out_deltas[o] * self.hiddens[h]
                self.hidden_weights[h][o] += ((change * learning) +
                                              (momentum * self.hidden_weight_overlay[h][o]))
                self.hidden_weight_overlay[h][o] = change

        # update input weights
        for i in xrange(self.num_inputs):
            for h in xrange(self.num_hidden):
                change = hidden_deltas[h] * self.inputs[i]
                self.in_weights[i][h] += ((change * learning) +
                                          (momentum * self.in_weight_overlay[i][h]))
                self.in_weight_overlay[i][h] = change
        pp.pprint(self.hidden_weights)
        # calculate error
        error = 0.0
        for o in xrange(self.num_output):
            error += 0.5*(targets[o] - self.outputs[o])**2.0
        return error





    def trainRP(self, dataset, itter, outfile,dmin=0.000001, dmax= 50.0, npos=1.2, nneg = 0.5):

        out_grad= np.zeros(self.num_output)
        hidden_grad = np.zeros(self.num_hidden)

        # gradients for weights
        prev_in_hid_grads = np.full((self.num_inputs, self.num_hidden), 0.0)
        prev_hid_out_grads = np.full((self.num_hidden, self.num_output), 0.0)

        #deltas
        prev_in_hid_deltas = np.full((self.num_inputs, self.num_hidden), 0.1)
        prev_hid_out_deltas = np.full((self.num_hidden, self.num_output), 0.1)
        print "epoc", "\t", "count correct", "\t", "percent correct", "\t", "MSE"
        for epoc in xrange(itter):
            count_correct = 0

            # set accum matricies to 0
            in_hid_grads = np.zeros((self.num_inputs, self.num_hidden))
            hid_out_grads = np.zeros((self.num_hidden, self.num_output))

            mserror = 0.0
            for example in dataset:
                results = self.run(example[0])
                cor = False in(example[1]==results[1])
                if(cor == False):
                    count_correct+=1

                for o in xrange(self.num_output):
                    mserror += 0.5*(example[1][o] - self.outputs[o])**2.0




                # calulate out deltas as in backprop
                for o in xrange(self.num_output):
                    out_grad[o] = example[1][o] - self.outputs[o]
                    out_grad[o] *= self.error_function(self.outputs[o])

                # calculate hidden deltas as in backprop
                for h in xrange(self.num_hidden):
                    error = 0.0
                    for o in xrange(self.num_output):
                        error += out_grad[o] * self.hidden_weights[h][o]
                    hidden_grad[h] = error * self.error_function(self.hiddens[h])

                # accumulate hidden weight gradients
                for h in xrange(self.num_hidden):
                    for o in xrange(self.num_output):
                      grad = out_grad[o] * self.hiddens[h]
                      hid_out_grads[h][o] += grad

                # accumulate in weight gradients
                for i in xrange(self.num_inputs):
                    for h in xrange(self.num_hidden):
                      grad = hidden_grad[h] * self.inputs[i]
                      in_hid_grads[i][h] += grad

            for i in xrange(self.num_inputs):
                for h in xrange(self.num_hidden):
                    if in_hid_grads[i][h] * prev_in_hid_deltas[i][h] > 0:
                        delta = min(prev_in_hid_deltas[i][h] * npos, dmax)
                        tmp = np.sign(in_hid_grads[i][h]) * delta
                        self.in_weights[i][h] += tmp

                    elif in_hid_grads[i][h] * prev_in_hid_deltas[i][h] < 0:
                        delta = max(prev_in_hid_deltas[i][h] * nneg, dmin)
                        self.in_weights[i][h] -= prev_in_hid_deltas[i][h]
                        in_hid_grads[i][h] = 0

                    else:
                        delta = prev_in_hid_deltas[i][h]
                        tmp = np.sign(in_hid_grads[i][h]) * delta
                        self.in_weights[i][h] += tmp
                    prev_in_hid_deltas[i][h] = delta
                    prev_in_hid_grads[i][h] = in_hid_grads[i][h]

            for h in xrange(self.num_hidden):
                for o in xrange(self.num_output):
                    if hid_out_grads[h][o] * prev_hid_out_grads[h][o] > 0:
                        delta = min(prev_hid_out_deltas[h][o] * npos, dmax)
                        tmp = np.sign(hid_out_grads[h][o]) * delta
                        self.hidden_weights[h][o] += tmp

                    elif hid_out_grads[h][o] * prev_hid_out_grads[h][o] < 0:
                        delta = max(prev_hid_out_deltas[h][o] * nneg, dmin)
                        self.hidden_weights[h][o] -= prev_hid_out_deltas[h][o]
                        hid_out_grads[h][o] = 0

                    else:
                        delta = prev_hid_out_deltas[h][o]
                        tmp = np.sign(hid_out_grads[h][o]) * delta
                        self.hidden_weights[h][o] += tmp
                    prev_hid_out_deltas[h][o] = delta
                    prev_hid_out_grads[h][o] = hid_out_grads[h][o]

            print epoc, "\t\t", count_correct, "\t\t",count_correct/float(len(dataset)), "\t\t",mserror/float(len(dataset))








    def trainBP(self, dataset, itter, filepath,learning_rate=0.7, momentum=0.5, verbose=False):
        """Backpropigation based training
        Arguments:
        dataset --
        itter --
        learning_rate --
        momentum --
        """
        for i in xrange(itter):
            error = 0.0
            count_correct = 0
            error_record = []
            for example in dataset:
                self.run(example[0])
                ex_error = self.backprop(example[1], learning_rate, momentum)
                error += ex_error
            error /= float(len(dataset))

            if verbose:
                print i, ":\t\t", 'error =', error

            error_record.append([i, error])
            if (error < 0.02):
                return



    def test(self, patterns, verbose=False):
        """testing function
        Arguments:
        patterns --
        verbose --
        """

        count =0
        correct = 0
        results = []
        for p in patterns:
            count+= 1
            res = self.run(p[0])
            if verbose:
                print p[0], '->', [ '%.2f' % elem for elem in res[0] ],'=' , res[1], '=', p[1]
            cor = False in(p[1]==res[1])
            if(cor == False):
                correct+=1

        results = [len(patterns), correct, float(correct)/float(len(patterns))]
        print results
        return results


def iris_holdout_bp(test_portion, hidden_nodes, fun_pair, len_rate, momentum, shuffle = False):
    """
    Simple holdout method: split data into 2
    sets, training and validation
    :param training_size:
    :return:
    """
    ann = Network(4, hidden_nodes, 3, 0, 1, fun_pair[0], fun_pair[1])

    if shuffle:
        np.random.shuffle(iris_data)

    working_set = iris_data
    test_size = int(test_portion * len(iris_data))

    validation_set = working_set[:test_size]
    training_set = working_set[test_size:]

    ann.trainBP(training_set, 500, len_rate, momentum, True)
    ann.test(validation_set, True)

def cancer_holdout_bp(test_portion, hidden_nodes, fun_pair, len_rate, momentum, shuffle = False):
    """
    Simple holdout method: split data into 2
    sets, training and validation
    :param training_size:
    :return:
    """
    ann = Network(30, hidden_nodes, 2, 0, 1, fun_pair[0], fun_pair[1])

    if shuffle:
        np.random.shuffle(cancer_data)

    working_set = cancer_data
    test_size = int(test_portion * len(cancer_data))

    validation_set = working_set[:test_size]
    training_set = working_set[test_size:]

    ann.trainBP(training_set, 500, len_rate, momentum, True)

    print "validation accuracy"
    ann.test(validation_set, False)

    print "training accuracy"
    ann.test(training_set, False)


def iris_test_RP():
    np.random.shuffle(iris_data)
    train = iris_data[:80]
    test = iris_data[80:]

    ann = Network(4,  6, 3, 0, 1, activation, error_function)
    ann.trainRP(train, 500)
    ann.test(test)


def iris_test_bp():
    np.random.shuffle(iris_data)
    train = iris_data[:80]
    test = iris_data[80:]

    ann = Network(4,  9, 3, 0, 1, activation, error_function)
    ann.trainBP(train, 5)
    ann.test(test)



def cancer_test():
    np.random.shuffle(cancer_data)
    train = cancer_data[:455]
    test = cancer_data[455:]
    ann = Network(30, 16, 2, 0, 1, activation, error_function)
    ann.trainRP(train, 500)
    ann.test(test)


def xor_test():
    data =[
        [[0,0],[0,1]],
        [[1,0],[1,0]],
        [[0,1],[1,0]],
        [[1,1],[0,1]],
    ]
    ann = Network(2, 4, 2, 0, 1, activation, error_function)
    ann.trainRP(data,1000, "../data/xortest1.csv")
    ann.test(data, verbose=True)


iris_test_RP()