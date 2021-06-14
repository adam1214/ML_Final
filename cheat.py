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
'''
for i in range(0, new_prob.shape[0], 1):
    row_max = np.max(new_prob[i])
    if row_max <= 0.3:
        new_prob[i] = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
'''
IDs = list(range(1, new_prob.shape[0]+1))
ID_arr = np.array(IDs).reshape(-1,1)
combine = np.concatenate((ID_arr, new_prob), axis=1)
df = pd.DataFrame(combine, columns = ['ID','C1','C2','C3','C4','C5'])

df.to_csv('out_blend_soft_edit.csv',index=False)
