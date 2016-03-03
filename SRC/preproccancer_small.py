#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def apply_label(row, dict):
    return dict[row['string_class']]



df = pd.read_csv("../data/wisc/wisccancer.csv")

labels = [2, 4]
bit_string = ['01', '10']


dicta = dict(zip(labels, bit_string))

new_df = df.copy(deep=True)
row = new_df.iterrows()

new_df['bit_label'] = df.apply (lambda row: apply_label (row, dicta),axis=1)

new_df.drop('id', axis=1, inplace=True)
new_df.drop('string_class', axis=1, inplace=True)
new_df.to_csv(path_or_buf='../data/wisc/smallcancer_post.csv', index=False)