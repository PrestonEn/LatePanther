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
import pandas as pd

current_milli_time = lambda: int(round(time.time() * 1000))

random.seed(current_milli_time())
pp = pprint.PrettyPrinter(indent=1)


def create_dataset(file, attrib_col_start, attrib_col_end, class_col, header=True):
    values =  np.loadtxt(file, delimiter=',',
                         usecols=range(attrib_col_start,attrib_col_end+1),
                         skiprows=1)

    labels =  np.loadtxt(file, delimiter=',', usecols=[class_col], skiprows=1, dtype=str)
    nums = []
    for string in labels:
        nums.append([int(n) for n in list(string)])

    return [list(a) for a in zip(values.tolist(), nums)]
iris_data = create_dataset("../data/iris/iris_post_norm_0to1.csv",0,3,4)
cancer_data = create_dataset("../data/wisc/cancer_post_norm_0to1.csv",0,29,30)
small_cancer = create_dataset("../data/wisc/smallcancer_post.csv",0,8,9)

def activation(x):
    return 1.0 / (1.0 + math.exp(-x))

def error_function(y):
    return y * (1.0 - y)

def tanh_error(y):
    return 1-y**2

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

        # rProp variables

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

        # update input overlay
        for i in xrange(self.num_inputs):
            for h in xrange(self.num_hidden):
                change = hidden_deltas[h] * self.inputs[i]
                self.in_weights[i][h] += ((change * learning) +
                                          (momentum * self.in_weight_overlay[i][h]))
                self.in_weight_overlay[i][h] = change

        # calculate error
        error = 0.0
        for o in xrange(self.num_output):
            error += 0.5*(targets[o] - self.outputs[o])**2.0
        return error

    def trainBP(self, dataset, itter, learning_rate, momentum, verbose=False):
        """Backpropigation based training
        Arguments:
        dataset --
        itter --
        learning_rate --
        momentum --
        """
        mse = []
        classi_error = []
        errors = [mse, classi_error]
        for i in xrange(itter):
            error = 0.0
            count_correct = 0

            for example in dataset:
                results = self.run(example[0])
                ex_error = self.backprop(example[1], learning_rate, momentum)
                error += ex_error
                cor = False in(example[1]==results[1])
                if(cor == False):
                    count_correct+=1

            error /= float(len(dataset))
            mse.append(error)
            classi_error.append(1.0 - count_correct/float(len(dataset)))

            if i % 10 == 0:
               print "epoc:\t", i, "\tmse:\t", error, "\t% wrong:\t", 1.0 - count_correct/float(len(dataset))
        return errors





    def trainRP(self, dataset, itter ,dmin=0.000001, dmax= 50.0, npos=1.2, nneg = 0.5, verbose=False):

        out_grad= np.zeros(self.num_output)
        hidden_grad = np.zeros(self.num_hidden)

        # gradients for weights
        prev_in_hid_grads = np.full((self.num_inputs, self.num_hidden), 0.0)
        prev_hid_out_grads = np.full((self.num_hidden, self.num_output), 0.0)

        #deltas
        prev_in_hid_deltas = np.full((self.num_inputs, self.num_hidden), 0.1)
        prev_hid_out_deltas = np.full((self.num_hidden, self.num_output), 0.1)


        mse = []
        classi_error = []

        errors = [mse, classi_error]
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

            if epoc % 10 == 0:
                print "epoc:\t", epoc, "\tmse:\t", mserror, "\t% wrong:\t", 1.0 - count_correct/float(len(dataset))
            mse.append(mserror)
            classi_error.append(1.0 - count_correct/float(len(dataset)))
        return errors



    def test(self, patterns, verbose=False):
        """testing function
        Arguments:
        patterns --
        verbose --
        """
        count =0
        correct = 0
        outcome = []
        for p in patterns:
            count+= 1
            res = self.run(p[0])

            print p[0], '->', [ '%.2f' % elem for elem in res[0] ],'=' , res[1], '=', p[1]
            cor = False in(p[1]==res[1])
            if(cor == False):
                outcome.append(True)
                correct+=1
            else:
                outcome.append(False)

        results = [float(correct)/float(len(patterns)), outcome]
        print results[0]
        return results


# def k_fold_bp(folds, anns,  learn_rates, momentums, filename):
#
#     models =[]
#     results = []
#     for i in xrange(len(anns)):
#         models.append([])
#         results.append([])
#
#     for k in xrange(folds):
#         for i in xrange(len(anns)):
#             models[i].append(copy.deepcopy(anns[i])


def iris_holdout_bp(test_portion,ann, learn_rate, momentum):
    """
    Simple holdout method: split data into 2
    sets, training and validation
    :param training_size:
    :return:
    """
    working_set = iris_data
    test_size = int(test_portion * len(iris_data))

    validation_set = working_set[:test_size]
    training_set = working_set[test_size:]

    train_errors = ann.trainRP(training_set, 200, False)
    test_errors = ann.test(validation_set, False)
    return test_errors

def iris_holdout_rp(test_portion,ann):
    """
    Simple holdout method: split data into 2
    sets, training and validation
    :param training_size:
    :return:
    """

    working_set = iris_data
    test_size = int(test_portion * len(iris_data))

    validation_set = working_set[:test_size]
    training_set = working_set[test_size:]

    train_errors = ann.trainRP(training_set, 200, False)
    test_errors = ann.test(validation_set, False)
    return test_errors

def cancer_holdout_bp(test_portion, ann, learn_rate, momentum):
    """
    Simple holdout method: split data into 2
    sets, training and validation
    :param training_size:
    :return:
    """
    working_set = small_cancer
    test_size = int(test_portion * len(cancer_data))

    validation_set = working_set[:test_size]
    training_set = working_set[test_size:]

    ann.trainBP(training_set, 200, learn_rate, momentum, True)
    ann.test(validation_set, False)

def cancer_holdout_rp(test_portion, ann):
    """
    Simple holdout method: split data into 2
    sets, training and validation
    :param training_size:
    :return:
    """
    working_set = small_cancer
    test_size = int(test_portion * len(cancer_data))

    validation_set = working_set[:test_size]
    training_set = working_set[test_size:]

    train_errors = ann.trainRP(training_set, 200, False)
    test_errors = ann.test(validation_set, False)
    print test_errors[0], "Validation"
    print ann.test(validation_set, False)
    return test_errors


# collect data for t-tests between models on cancer data
# different activation functions
# 80/20 holdout
rp_t_test = pd.DataFrame(index=np.arange(10), columns=["moda", "modb"])
model_a_test = []
model_b_test = []
for i in xrange(10):
    print i
    np.random.shuffle(small_cancer)
    model_a = Network(4,4,3,0,1,activation, error_function)
    model_b = Network(4,3,3,0,1,activation, error_function)
    out_a = iris_holdout_bp(0.3 ,model_a, 0.07, 0.5)
    out_b = iris_holdout_bp(0.3 ,model_b, 0.000001, 0.5)
    model_a_test.append(out_a[0])
    model_b_test.append(out_b[0])

rp_t_test['moda'] = model_a_test
rp_t_test['modb'] = model_b_test

rp_t_test.to_csv(path_or_buf='../data/bp_actv_t_test_iris_hid_nodes.csv', index=False)





