# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.metrics import f1_score

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.
    title : str
        Title for the chart.
    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.
    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.
    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.
    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes.legend(loc="best")

    return plt

if __name__ == "__main__":
    df_train = pd.read_csv('Datasets/norm/train.csv')
    #df_train_X = df_train.drop(labels=["Unnamed: 0","0","1","26"], axis=1)
    df_train_X = df_train.drop(labels=["ID","TS","Y"], axis="columns")
    df_train_Y = df_train['Y'].to_frame()

    df_valid = pd.read_csv('Datasets/norm/valid.csv')
    df_valid_X = df_valid.drop(labels=["ID","TS","Y"], axis="columns")
    df_valid_Y = df_valid['Y'].to_frame()


    svm = OneVsRestClassifier(SVC(kernel='rbf', probability=True, random_state=0, C=5.0), n_jobs=10)
    #LCV = SelectFromModel(LassoCV(cv=5), prefit=False, threshold=0.1)
    #svm = Pipeline([('feature_selection', LCV), ('classification', svm)])
    '''
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    fig, axes = plt.subplots(1, 1, figsize=(10, 15))
    title = r"Learning Curves (SVM)"
    plot_learning_curve(svm, title, df_train_X_std, df_train_Y['Y'].values, axes=axes, ylim=None, cv=cv, n_jobs=14)
    plt.show()
    '''
    svm.fit(df_train_X, df_train_Y['Y'].values)
    #print(df_train_X.columns[LCV.get_support()])
    y_pred_prob = svm.predict_proba(df_valid_X)
    predict = np.argmax(y_pred_prob, axis=1)
    for i in range(0, len(predict), 1):
        predict[i] = predict[i] + 1
    # predict = svm.predict(df_valid_X)
    ground_true = df_valid_Y['Y'].values
    
    error = 0
    for i, v in enumerate(predict):
        if v != ground_true[i]:
            error+=1
    print('ACC:', (2063-error)/2063)

    print('Log Loss:', log_loss(ground_true, y_pred_prob))
    
    f1 = f1_score(ground_true, predict, average='macro')
    print('F1 Score:', f1)