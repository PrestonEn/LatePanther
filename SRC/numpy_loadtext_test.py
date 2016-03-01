import numpy as np


def create_iris_data():
    values =  np.loadtxt("../data/iris/iris_post.csv", delimiter=',', usecols=[0,1,2,3], skiprows=1)
    labels =  np.loadtxt("../data/iris/iris_post.csv", delimiter=',', usecols=[4], skiprows=1, dtype=str)
    nums = []
    for string in labels:
        nums.append([int(n) for n in list(string)])

    return [list(a) for a in zip(values.tolist(), nums)]


def create_cancer_data():
    values =  np.loadtxt("../data/wisc/cancer_post_norm_0to1.csv", delimiter=',', usecols=range(0,30), skiprows=1)
    labels =  np.loadtxt("../data/wisc/cancer_post_norm_0to1.csv", delimiter=',', usecols=[30], skiprows=1, dtype=str)
    nums = []

    for string in labels:
        nums.append([int(n) for n in list(string)])

    return [list(a) for a in zip(values.tolist(), nums)]
