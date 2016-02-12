#!/usr/bin/python
# -*- coding: utf-8 -*-



import math
import numpy
import random


"""
BEGIN EXPERIMENT VALS
"""
random.seed(666)


class Network(object):
    """ Neural network
    Arguments:
    inputs -- number of input neurons
    hidden -- number of hidden layer neurons
    outputs -- number of output classes
    w_min -- minimum random range for generating initial weights
    w_max -- maximum random range for generating initial weights


    """
    def __init__(self, inputs, hidden, outputs, w_min, w_max, act_der, bias=1):
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
        self.outs = numpy.ones(self.num_out)

        # initialize the weight matrices to 0
        self.in_weights = numpy.random.uniform(w_min,
                                               w_max,
                                               (self.num_inputs,
                                                self.num_hidden))

        self.hidden_weights = numpy.random.uniform(w_min,
                                                   w_max,
                                                   (self.num_hidden,
                                                    self.num_output))

        # build weight update overlay matrices
        self.in_weight_overlay = numpy.zeros((self.num_inputs,
                                              self.num_hidden))

        self.hidden_weight_overlay = numpy.zeros((self.num_hidden,
                                                  self.num_output))
    # def __init__

    # network is initialized, now what?

    # run(self, file of inputs)
        # take array of attributes and set input nodes,
        # turn bitstring class into array and set outputs

        # do it, multiply inputs by weights to hidden,
        # pass sum to activation function, go go go


    # backprop(self, the target values, learning rate, momentum)


    # train

    # test

    # output

