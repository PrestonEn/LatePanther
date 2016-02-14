#!/usr/bin/python
# -*- coding: utf-8 -*-



import math
import numpy
import random
from math import tanh

"""
BEGIN EXPERIMENT VALS
"""
random.seed(666)

def tanhdir(y):
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

    def run(self, inputs):
        # set inputs
        for i in xrange(self.num_inputs - self.bias):
            self.inputs[i] = inputs[i]

        # pass inputs to hiddens
        for h in xrange(self.num_hidden - self.bias):
            total = 0.0
            for i in xrange(self.num_inputs):
                total += self.inputs[i] * self.in_weights[i][h]
            self.hiddens[h] = tanh(total)

        # pass hiddens to outputs
        for o in xrange(self.num_output):
            total = 0.0
            for h in xrange(self.num_hidden):
                total += self.hiddens[h] * self.hidden_weights[h][o]
            self.outputs[o] = tanh(total)
        return self.outputs[:]

    """Backpropigation
    Arguments:
    targets -- array of target values of form [0,0,1]
    learn_rt -- 0.0 < value <= 1.0, learning rate
    momentum -- 0.0 < value <= 1.0
    """
    def backprop(self, targets, learning, momentum):
        out_deltas = numpy.zeros(self.num_output)
        hidden_deltas = numpy.zeros(self.num_hidden)

        # calulate out deltas
        for o in xrange(self.num_output):
            out_deltas[o] = targets[o] - self.outputs[o]
            out_deltas[o] *= tanhdir(self.outputs[o])

        # calculate hidden deltas
        for h in xrange(self.num_hidden):
            error = 0.0
            for o in xrange(self.num_output):
                error += out_deltas[o] * self.hidden_weights[h][o]
            hidden_deltas[h] = error * tanhdir(self.hiddens[h])

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




    """Train
    """
    def train(self, dataset, itter, learning_rate, momentum):
        # give a training example
        # do the forward pass
        # run backprop
        for i in xrange(itter):
            error = 0.0
            for example in dataset:
                self.run(example[0])
                ex_error = self.backprop(example[1], learning_rate, momentum)
                error += ex_error
            error /= float(len(dataset))
            print error


    """test
    """
    def test(self, patterns, verbose=False):
        tmp = []

        for p in patterns:
            if verbose:
                print p[0], '->', self.run(p[0]), '=', p[1]
            tmp.append(self.run(p[0]))
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
xor_test()
