# Python version cannot be 3.8
import sys
print(sys.version)

from alexandria.experiment import Experiment, Experiments
import pandas as pd

print('Imports complete.')

datapath = '/media/notclaytonjohnson/Seagate Portable Drive/Data/mooc/kdd_all_normalized_features.csv'

df = pd.read_csv(datapath)
df.head()

X_cols = list(df.columns)
X_cols.remove('enrollment_id')
X_cols.remove('truth')
y_col = 'truth'

X_cols

# Create the KDD experiment object
kdd_exp = Experiment(
    name='KDD Dataset Experiments',
    dataset=df,
    xlabels=X_cols,
    ylabels=y_col,
    models=[
        'rf', 
        'dt', 
        'knn',
        'nb',
        'ab',
        'gb',
        'xgb',
        'lr',
        'da'
    ]
)

datapath = '/media/notclaytonjohnson/Seagate Portable Drive/Data/mooc/stanford_normalized_all_features.csv'

df = pd.read_csv(datapath)
df.head()

X_cols = list(df.columns)
X_cols.remove('enrollment_id')
X_cols.remove('label')
y_col = 'label'

X_cols

# Create the Stanford experiment object
stan_exp = Experiment(
    name='Stanford Dataset Experiments',
    dataset=df,
    xlabels=X_cols,
    ylabels=y_col,
    models=[
        'rf', 
        'dt', 
        'knn',
        'nb',
        'ab',
        'gb',
        'xgb',
        'lr',
        'da'
    ]
)

datapath = '/media/notclaytonjohnson/Seagate Portable Drive/Data/mooc/xeutangx_all_normalized_features.csv'

df = pd.read_csv(datapath)
df.head()

X_cols = list(df.columns)
X_cols.remove('enroll_id')
X_cols.remove('truth')
y_col = 'truth'

X_cols

# Create the Xeutangx experiment object
xeu_exp = Experiment(
    name='Xeutangx Dataset Experiments',
    dataset=df,
    xlabels=X_cols,
    ylabels=y_col,
    models=[
        'rf', 
        'dt', 
        'knn',
        'nb',
        'ab',
        'gb',
        'xgb',
        'lr',
        'da'
    ]
)

exps = Experiments()
exps.addExperiment(kdd_exp)
exps.addExperiment(stan_exp)
exps.addExperiment(xeu_exp)

exps.trainCV(nfolds=10, metrics=['acc', 'prec', 'rec', 'auc'], reportduring=True)
