from sklearn.datasets import load_iris
import numpy as np
import os
import pandas as pd

from alexandria.experiments import Experiment

classifiers = ['rf', 'xgb', 'ab', 'dt', 'knn', 'svm', 'lr', 'lda', 'nb']

def KDDExperiments():
    path = 'data/kddcup15'
    db_path = os.path.join(path, 'kdd_all_normalized_features.csv')
    df = pd.read_csv(db_path)
    # Features that could lead to overfitting the models
    bad_features = ['enrollment_id']
    df.drop(labels=bad_features, axis='columns', inplace=True)
    dep_var = 'truth'
    X = df.loc[:, df.columns != dep_var]
    y = df[dep_var]
    experiment = Experiment('MOOC Dropout Prediction - KDD', models=classifiers, 
                    metrics=['acc', 'rec', 'prec'],
                    exp_type='classification')
    """experiment.train(
        X, 
        y, 
        metrics=['acc', 'rec', 'prec'], 
        cv=True, 
        n_folds=3, 
        shuffle=True)
    """
    #experiment.summarizeMetrics()
    #experiment.compareModels_2x5cv(models=['rf', 'dt', 'ab'], X=X, y=y) # models='all'
    experiment.compareModels_ttest(X=X, y=y)


def xeutangx_Experiments():
    path = 'data/xeutangx'

    # Import the training data
    f_train = 'train_normalized_trimmed_features.csv'
    df_train = pd.read_csv(os.path.join(path, f_train))

    f_test = 'test_normalized_trimmed_features.csv'
    df_test = pd.read_csv(os.path.join(path, f_test))
    bad_features = ['enroll_id', 'username', 'course_id']
    df_train.drop(labels=bad_features, axis='columns', inplace=True)
    df_test.drop(labels=bad_features, axis='columns', inplace=True)
    df = pd.concat([df_train, df_test])
    dep_var = 'truth'
    X = df.loc[:, df.columns != dep_var]
    y = df[dep_var]
    experiment = Experiment('MOOC Dropout Prediction - Xeutang-X', models=classifiers, 
                    metrics=['acc', 'rec', 'prec'],
                    exp_type='classification')
    experiment.compareModels_ttest(X=X, y=y)

def Stanford_KDD():
    path = 'data/stanford'
    db_path = os.path.join(path, 'normalized_all_features.csv')
    df = pd.read_csv(db_path, index_col=0)
    # Features that could lead to overfitting the models
    bad_features = ['enrollment_id']
    df.drop(labels=bad_features, axis='columns', inplace=True)
    dep_var = 'label'
    X = df.loc[:, df.columns != dep_var]
    y = df[dep_var]

    experiment = Experiment('MOOC Dropout Prediction - Xeutang-X', models=classifiers, 
                    metrics=['acc', 'rec', 'prec'],
                    exp_type='classification')
    experiment.compareModels_ttest(X=X, y=y)

def test():
    #__package__ == 'alexandria'

    # Data preprocessing
    print('With sklearn DataBunch object...')
    iris = load_iris()
    X, y = iris.data, iris.target

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    experiment = Experiment('iris example experiment', models=['rf', 'dt', 'ab', 'knn'], exp_type='classification', metrics=['acc', 'rec', 'prec'])
    experiment.train(
        X, 
        y,  
        cv=True, 
        n_folds=10, 
        shuffle=True)
    experiment.summarizeMetrics()

    print('\n')

    # Conduct pairwise comparison of scores for models
    experiment.compareModels_2x5cv(models='all', X=X, y=y)

    print('\n')


    # Data preprocessing for dataframe object
    print('With pandas DataFrame object...')
    iris_df = load_iris(as_frame=True).frame
    X = iris_df.loc[:, iris_df.columns != 'target']
    y = iris_df['target']
    
    experiment = Experiment('iris example experiment', models=['rf', 'dt', 'ab', 'knn'], exp_type='classification')
    experiment.train(
        X, 
        y, 
        metrics=['acc', 'rec', 'prec'], 
        cv=True, 
        n_folds=10, 
        shuffle=True)
    experiment.summarizeMetrics()

if __name__ == '__main__':
    print('running experiments')
    #KDDExperiments()
    #xeutangx_Experiments()
    Stanford_KDD()
