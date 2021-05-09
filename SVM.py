import csv
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

if __name__ == "__main__":
    df_train = pd.read_csv('train.csv')
    df_train_X = df_train.drop(labels=["ID","TS","Y"], axis="columns")
    df_train_Y = df_train['Y'].to_frame()

    df_valid = pd.read_csv('valid.csv')
    df_valid_X = df_valid.drop(labels=["ID","TS","Y"], axis="columns")
    df_valid_Y = df_valid['Y'].to_frame()
    
    sc = StandardScaler()
    sc.fit(df_train_X)
    df_train_X_std = sc.transform(df_train_X)
    df_valid_X_std = sc.transform(df_valid_X)

    svm = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    svm.fit(df_train_X_std, df_train_Y['Y'].values)

    predict = svm.predict(df_valid_X_std)
    ground_true = df_valid_Y['Y'].values

    
    error = 0
    for i, v in enumerate(predict):
        if v != ground_true[i]:
            error+=1
    print(error/2063)
