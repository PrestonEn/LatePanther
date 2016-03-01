#!/usr/bin/python
# -*- coding: utf-8 -*-
import pprint
import math
import numpy
import random
import numpy_loadtext_test as loadd
import pandas as pd
import csv

random.seed(4564)
pp = pprint.PrettyPrinter(indent=4)


def activation(x):
    return 1.0 / (1.0 + math.exp(-x))

def error_function(y):
    return y * (1.0 - y)

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
    def __init__(self, inputs, hidden, outputs, w_min, w_max, bias=1):
        # if bias = 1, we have a bias on the input-hidden
        # and hidden-output matrix
        self.bias = bias

        # add bias to input and hidden layers
        self.num_inputs = inputs + self.bias
        self.num_hidden = hidden + self.bias
        self.num_output = outputs

        # initialize the layer value lists
        self.inputs = numpy.ones(self.num_inputs)
        self.hiddens = numpy.ones(self.num_hidden)
        self.outputs = numpy.ones(self.num_output)

        # initialize the weight matrices to 0
        self.in_weights = numpy.random.uniform(w_min, w_max,
                                               (self.num_inputs,
                                                self.num_hidden))

        self.hidden_weights = numpy.random.uniform(w_min, w_max,
                                                   (self.num_hidden,
                                                    self.num_output))

        # build weight update overlay matrices
        self.in_weight_overlay = numpy.zeros((self.num_inputs,
                                              self.num_hidden))

        self.hidden_weight_overlay = numpy.zeros((self.num_hidden,
                                                  self.num_output))

        # rProp variables



    def run(self, inputs):
        """
        Arguments:
        targets -- array of target values of form [0,0,1]
        learn_rt -- 0.0 < value <= 1.0, learning rate
        momentum -- 0.0 < value <= 1.0
        """
        # set inputs
        for i in xrange(self.num_inputs - self.bias):
            self.inputs[i] = inputs[i]

        # pass inputs to hiddens
        for h in xrange(self.num_hidden - self.bias):
            total = 0.0
            for i in xrange(self.num_inputs):
                total += self.inputs[i] * self.in_weights[i][h]
            self.hiddens[h] = activation(total)

        # pass hiddens to outputs
        for o in xrange(self.num_output):
            total = 0.0
            for h in xrange(self.num_hidden):
                total += self.hiddens[h] * self.hidden_weights[h][o]
            self.outputs[o] = activation(total)

        # map the results to a more readable predicted class
        max_value = max(self.outputs)
        max_index = self.outputs.tolist().index(max_value)
        predicted_class = numpy.zeros(len(self.outputs))
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
        out_deltas = numpy.zeros(self.num_output)
        hidden_deltas = numpy.zeros(self.num_hidden)

        # calulate out deltas
        for o in xrange(self.num_output):
            out_deltas[o] = targets[o] - self.outputs[o]
            out_deltas[o] *= error_function(self.outputs[o])

        # calculate hidden deltas
        for h in xrange(self.num_hidden):
            error = 0.0
            for o in xrange(self.num_output):
                error += out_deltas[o] * self.hidden_weights[h][o]
            hidden_deltas[h] = error * error_function(self.hiddens[h])

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



    def train(self, dataset, itter, learning_rate, momentum):
        """training function
        Arguments:
        dataset --
        itter --
        learning_rate --
        momentum --
        """
        for i in xrange(itter):
            error = 0.0
            count_correct = 0
            for example in dataset:
                self.run(example[0])
                ex_error = self.backprop(example[1], learning_rate, momentum)
                error += ex_error
            error /= float(len(dataset))

            print i, ":\t\t", 'error =', error
            if (error < 0.02):
                return


    def test(self, patterns, verbose=False):
        """testing function
        Arguments:
        patterns --
        verbose --
        """
        tmp = []
        count =0
        correct = 0
        for p in patterns:
            res = self.run(p[0])
            if verbose:
                print p[0], '->', [ '%.2f' % elem for elem in res[0] ], '=', p[1] , '=' , res[1]
            count+= 1
            cor = False in(p[1]==res[1])
            if(cor == False):
                correct+=1

        print count, "    ", correct, "    ", float(correct)/float(count)
        return tmp

def xor_test():
    data =[
        [[0,0],[0,1]],
        [[1,0],[1,0]],
        [[0,1],[1,0]],
        [[1,1],[0,1]],
    ]
    ann = Network(2,4,2,-1,1)
    ann.train(data,1000,0.5,0.5)
    ann.test(data, verbose=True)



def iris_test():
    data = loadd.create_iris_data()
    numpy.random.shuffle(data)


    train = data[:80]
    test = data[80:]

    pp.pprint(train)

    ann = Network(4,  9, 3, -1, 1)
    ann.train(train, 500,0.7,0.5)
    ann.test(test, verbose=True)


def cancer_test():
    data = loadd.create_cancer_data()
    numpy.random.shuffle(data)

    train = data[:300]
    test = data[300:]

    pp.pprint(train)

    ann = Network(30,  8, 2, 0, 1)
    ann.train(train, 1000,0.7,0.5)
    ann.test(test, verbose=True)


cancer_test()