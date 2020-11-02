# alexandria
This is a high-level machine learning framework that allows for the users to easily run multiple types of machine learning experiments at the drop of a hat. I'm currently working on developing this project, along with the [wiki pages](https://github.com/JohnsonClayton/alexandria/wiki) further.


An example for the API is below:

```python
# main.py - DataBunch and DataFrame demonstrations
# Data preprocessing
print('With sklearn DataBunch object...')
iris = load_iris()
X, y = iris.data, iris.target

experiment = Experiment('iris example experiment', models=['rf', 'dt', 'ab', 'knn'], exp_type='classification')
experiment.train(
    X, 
    y, 
    metrics=['acc', 'rec', 'prec'], 
    cv=True, 
    n_folds=10, 
    shuffle=True)
experiment.summarizeMetrics()

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
```
```
With sklearn DataBunch object...
model    acc            rec            prec
-------  -------------  -------------  -------------
rf       0.9400±0.0467  0.9400±0.0467  0.9532±0.0341
dt       0.9400±0.0554  0.9400±0.0554  0.9542±0.0362
ab       0.9267±0.0554  0.9267±0.0554  0.9414±0.0447
knn      0.9533±0.0306  0.9533±0.0306  0.9611±0.0255
With pandas DataFrame object...
model    acc            rec            prec
-------  -------------  -------------  -------------
rf       0.9400±0.0467  0.9400±0.0467  0.9532±0.0341
dt       0.9400±0.0554  0.9400±0.0554  0.9542±0.0362
ab       0.9267±0.0554  0.9267±0.0554  0.9414±0.0447
knn      0.9533±0.0306  0.9533±0.0306  0.9611±0.0255
```
