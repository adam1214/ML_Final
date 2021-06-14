# -*- coding: utf-8 -*-
"""
Created on Sun May 23 04:27:02 2021

@author: world Danny
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

read_path = './out_blend_soft.csv'

xy = np.loadtxt(read_path, delimiter = ',',dtype = np.str, skiprows = 1)
prob = xy[:,1:].astype(np.float32)

new_prob = np.zeros_like(prob)
for i in range(prob.shape[0]):
    if any(prob[i] > 0.72) :
        big = np.argmax(prob[i])
        new_prob[i][big] = 1
    else:
        new_prob[i] = prob[i]
        
for i in range(prob.shape[0]):
    for j in range(prob.shape[1]):
        if prob[i][j] > 0.71 and prob[i][j] < 0.72:
            print(i,j)