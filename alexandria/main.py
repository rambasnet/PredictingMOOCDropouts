from sklearn.datasets import load_iris
import numpy as np

from experiments import Experiment

if __name__ == '__main__':
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