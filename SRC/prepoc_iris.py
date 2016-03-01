#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def apply_label(row, dict):
    return dict[row['string_class']]

def normalize(row, name, min, max):
    return row[name] - min/(max - min)


df = pd.read_csv("../data/iris/iris.csv")
labels =  ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
bit_string = ['001', '010', '100']

dicta = dict(zip(labels, bit_string))

new_df = df.copy(deep=True)
row = new_df.iterrows()

maxes =  [df['A'].max(), df['B'].max(), df['C'].max(), df['D'].max()]
mins =  [df['A'].min(), df['B'].min(), df['C'].min(), df['D'].min()]

new_df['bit_label'] = df.apply (lambda row: apply_label (row, dicta),axis=1)
# new_df['Anorm'] = df.apply (lambda row: normalize(row, 'A', mins[0], maxes[0]),axis=1)
# new_df['Bnorm'] = df.apply (lambda row: normalize(row, 'B', mins[0], maxes[0]),axis=1)
# new_df['Cnorm'] = df.apply (lambda row: normalize(row, 'C', mins[0], maxes[0]),axis=1)
# new_df['Dnorm'] = df.apply (lambda row: normalize(row, 'D', mins[0], maxes[0]),axis=1)


new_df.drop('string_class', axis=1, inplace=True)
# new_df.drop('A', axis=1, inplace=True)
# new_df.drop('B', axis=1, inplace=True)
# new_df.drop('C', axis=1, inplace=True)
# new_df.drop('D', axis=1, inplace=True)
new_df.to_csv(path_or_buf='../data/iris/iris_post.csv', index=False)

