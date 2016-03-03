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

current_milli_time = lambda: int(round(time.time() * 1000))

random.seed(current_milli_time())
pp = pprint.PrettyPrinter(indent=1)


def create_dataset(file, attrib_col_start, attrib_col_end, class_col, header=True):
    values =  np.loadtxt(file, delimiter=',',
                         usecols=range(attrib_col_start,attrib_col_end+1),
                         skiprows=1, dtype=int)

    labels =  np.loadtxt(file, delimiter=',', usecols=[class_col], skiprows=1, dtype=str)
    # nums = []
    # for string in labels:
    #     nums.append([int(n) for n in list(string)])
    #
    # return [list(a) for a in zip(values.tolist(), nums)]

small_cancer = create_dataset("data/wisc/smallcancer_post.csv",0,8,9)