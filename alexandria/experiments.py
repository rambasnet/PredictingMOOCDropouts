from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score

import numpy as np
import pandas as pd
from mlxtend.evaluate import paired_ttest_5x2cv 

from tabulate import tabulate

if __package__ != 'alexandria':
    from utils import Helper
    from model import Model
    from metric import MetricsManager, Metric
else:
    from .utils import Helper
    from .model import Model
    from .metric import MetricsManager, Metric

class Experiments:
    '''
    Experiments object keeps track of all experiment objects provided
    '''
    def __init__(self, experiments=[]):
        # Keep track of all of the experiment objects for this experiment
        self.experiment_dict = {}

        if type(experiments) == list and len(experiments) > 0:
            # TO-DO: Should we check if these are legitimate experiment objects? - Probably
            for exp in experiments:
                self.addExperiment(exp)

        # Keep constants around
        self.default_experiment_name = 'experiment_'
        self.num_of_experiments = len(self.experiment_dict)

    def getDefaultExperimentName(self):
        return self.default_experiment_name + str(self.num_of_experiments)

    def addExperiment(self, experiment):
        if type(experiment) == Experiment:
            exp_name = experiment.getName()
            if exp_name == 'unnamed experiment':
                # Set its name to the default name. Do NOT change the experiment object's name
                exp_name = self.getDefaultExperimentName() 

            # Set the experiment as the value to it's own name
            self.experiment_dict[exp_name] = experiment
        else:
            raise ValueError('Object must be Experiment object: {}'.format(str(experiment)))

    def runAllExperiments(self):
        if len(self.experiment_dict.keys()) > 0:
            for experiment in self.experiment_dict.keys():
                experiment.train()
        else:
            raise ValueError('Experiments object has no models to run!')

    def getNumOfExperiments(self):
        return len(self.experiment_dict)

    def getExperiments(self):
        return self.experiment_dict

    def getExperimentNames(self):
        return list(self.experiment_dict.keys())


class Experiment:
    def __init__(self, name='', models=[], exp_type=None, metrics=[]):
        self.helper = Helper()
        self.unnamed_model_count = 0
        self.random_state = 0
        self._valid_experiment_types = ['classification', 'regression']

        self.metrics_manager = MetricsManager()
    
        self.type_of_experiment = None
        if exp_type:
            if type(exp_type) != str:
                raise ValueError('Experiment type attribute must be string, not {}'.format( str(type(exp_type)) ))

            if self.isValidExperimentType(exp_type):
                self.type_of_experiment = exp_type
            else:
                raise ValueError('The provided experiment type is not supported: \'{}\'\nOnly these experiment types are supported: {}'.format( exp_type, self.getValidExperimentTypes() ))
            

        if type(name) is str:
            if name != '':
                self.name = name
            else:
                self.name = 'unnamed_experiment'
        else:
            raise ValueError('Experiment name attribute must be string, not {}'.format(str( type(name) )))

        # initialize the dictionary of models within the experiment
        self.models_dict = {}

        # Add the models (if any) that the user wants to use
        if type(models) is list:
            if len(models) > 1:
                # Then add all of the model objects to the experiment
                for model in models:
                    self.addModel(model)
            elif len(models) == 1:
                # Add the only model to the experiment list
                self.addModel(models)
        elif type(models) is dict:
            # Add all of the models with their corresponding names provided from the user
            for name in models.keys():
                self.addModel(model, name)
        elif type(models) is str:
            self.addModel(models)
        else:
            raise ValueError('Models must be in list format if more than one is provided. Ex. models=[\'rf\', \'Decision Tree\', RandomForestClassifer()... ]')

        # initialize the metrics list
        self.metrics = []

        if type(metrics) is list:
            self.metrics = metrics
        elif type(metrics) is str:
            self.metrics = [metrics]
        else:
            raise ValueError('\'metrics\' argument must be string or list of strings. Ex: [\'accuracy\', \'prec\']. Cannot be {}'.format( str(type(metrics)) ))

    def getDefaultModelName(self):
        return 'model_' + str(len(self.models_dict))

    def getValidExperimentTypes(self):
        return self._valid_experiment_types

    def isValidExperimentType(self, exp_type):
        if type(exp_type) == str:
            if exp_type in self._valid_experiment_types:
                return True
            else:
                return False
        else:
            raise ValueError('Experiment type must be string: {}'.format(str(exp_type)))

    def setExperimentType(self, exp_type):
        if self.isValidExperimentType(exp_type):
            self.type_of_experiment = exp_type
        else:
            raise ValueError('Experiment must be \'regression\' or \'classification\', cannot be {}'.format(str(exp_type)))

    def getExperimentType(self):
        return self.type_of_experiment

    def getDefaultVersionOf(self, model_name):
        # TO-DO: Contact state object to get the object!
        # Do we know what type of experiment we're dealing with here?
        if self.type_of_experiment:
            # Figure out what type of model we need
            if self.type_of_experiment == 'regression':
                regularized_model_name = model_name + ':regressor'
            elif self.type_of_experiment == 'classification':
                regularized_model_name = model_name + ':classifier'
            return Model( model_name, self.helper.getDefaultModel(regularized_model_name), self.helper.getDefaultArgs(regularized_model_name) ) 
        else:
            return Model( model_name )

    def addModel(self, model, name=''):
        # TO-DO: What are valid objects to pass here? String? Yes, but gives default. Actual model object (such as RandomForestClassifier)? Yes!
        if type(model) is str:
            # Add the default version of the model they are requesting
            if model != '':
                model_name = model
                model = self.getDefaultVersionOf(model_name)
                self.models_dict[model_name] = model
            else:
                if self.name:
                    raise ValueError('Experiment object {} cannot add model default model: {}'.format( str(self.name), model))
                else:
                    raise ValueError('Experiment object cannot add model default model: {}'.format(model))
        elif type(model) is Model:
            model_name = model.getName()
            if model_name:
                self.models_dict[model_name] = model
            elif type(name) == str and name != '':
                self.models_dict[name] = model
            else:
                self.models_dict[self.getDefaultModelName()] = model
        elif type(model) is object:
            # TO-DO: What type of object is it? Scikit learn?
            print('the model is some kind of object!')

        # TO-DO: Find a way to generate automatic names for these models in a way that sounds smarter than model_1, model_2, ...
        elif type(name) is str:
            if name == '':
                # Default name will be assigned
                name = self.getDefaultModelName()
            elif self.hasModelByName(name):
                raise ValueError('Experiment object already has model by name: {}'.format(name))
        else:
            raise ValueError('Model name must be string: {}'.format(str(name)))

    def train(self, X, y, cv=False, n_folds=0, shuffle=False, metrics=[]):
        if len(X) == len(y):
            if cv:
                if n_folds > 0:
                    # If they want a ROC curve, start that
                    if 'roc' in metrics or 'ROC' in metrics or 'roc' in self.metrics or 'ROC' in self.metrics:
                        self.metrics_manager.startROCCurve()
                        print('startedROCCurve')

                    # Cross validation
                    sss = None
                    if shuffle:
                        sss = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
                    else:
                        sss = StratifiedKFold(n_splits=n_folds)
                    fold_num = 0
                    for train_idx, val_idx in sss.split(X, y):
                        fold_num += 1

                        # Organize the data
                        X_train, y_train = [], []
                        X_val, y_val = [], []
                        if type(X) == np.ndarray and type(y) == np.ndarray:
                            X_train, y_train = X[train_idx], y[train_idx]
                            X_val, y_val = X[val_idx], y[val_idx]
                        elif type(X) == pd.DataFrame and (type(y) == pd.Series or type(y) == list):
                            X_train, y_train = X.iloc[train_idx], y[train_idx]
                            X_val, y_val = X.iloc[val_idx], y[val_idx]
                        else:
                            raise ValueError('unrecognized datatypes:\n\ttype(X) => {}\n\ttype(y) => {}'.format( str(type(X)), str(type(y)) ))

                        for model_name in self.models_dict.keys():
                            model = self.models_dict[model_name]
                            model.run(X_train, y_train)

                        if metrics:
                            # Evaluate the performance of this model and keep track of it
                            self.eval(X_val, y_val, metrics=metrics, fold_num=fold_num)
                        elif len(self.metrics) > 0:
                            # Evaluate the performance of this model and keep track of it
                            self.eval(X_val, y_val, metrics=self.metrics, fold_num=fold_num)
                    if ('roc' in metrics or 'ROC' in metrics) and fold_num == n_folds:
                        print('reached')
                        self.metrics_manager.showROCCurve()
                else:
                    raise ValueError('Number of folds in cross validation (n_folds) must be larger than zero!')
            else:
                for model_name in self.models_dict.keys():
                    model = self.models_dict[model_name]
                    model.run(X, y)
            self.completed = True
        else:
            if self.name:
                raise ValueError('Data and target provided to \'{}\' must be same length:\n\tlen of data: {}\n\tlen of target: {}'.format(self.name, str(len(X)), str(len(y))))
            else:
                raise ValueError('Provided data and target must be same length:\n\tlen of data: {}\n\tlen of target: {}'.format(str(len(X)), str(len(y))))

    def predict(self, X):
        predicted_values = {}
        for model_name in self.models_dict.keys():
            model = self.models_dict[model_name]
            predicted_values[model_name] = model.predict(X)
        return predicted_values

    def eval(self, X, y, metrics=[], fold_num=None):
        if len(X) == len(y):
            # If we aren't running cross validation, then clear out the metrics
            if not fold_num:
                self.metrics_manager.clearMetrics()
            for model_name in self.models_dict.keys():
                model = self.models_dict[model_name]
                if fold_num and type(fold_num) == int:
                    metric_object = Metric(model_name, fold=fold_num)
                else:
                    metric_object = Metric(model_name)

                # If a ROC curve is requested, graph it!
                if 'roc' in metrics or 'ROC' in metrics:
                    fpr, tpr, roc_auc = model.calcROC(X, y, self.type_of_experiment)
                    self.metrics_manager.addToROCCurve(model_name, fpr, tpr, roc_auc)

                model_metrics = model.eval(X, y, metrics)
                for measure_name in model_metrics.keys():
                    metric_object.addValue(measure_name, model_metrics[measure_name])
                self.metrics_manager.addMetric(metric_object)

            if not fold_num:
                return self.metrics_manager.getMetrics()
        else:
            if self.name:
                raise ValueError('Data and target provided to \'{}\' must be same length:\n\tlen of data: {}\n\tlen of target: {}'.format(self.name, str(len(X)), str(len(y))))
            else:
                raise ValueError('Provided data and target must be same length:\n\tlen of data: {}\n\tlen of target: {}'.format(str(len(X)), str(len(y))))
        
    def getMetrics(self):
        return self.metrics_manager.getMetrics()

    def getSummarizedMetrics(self):
        return self.metrics_manager.printMeasures()

    def summarizeMetrics(self):
        metrics = self.metrics_manager.printMeasures()
        metrics_list_format = []

        headers = ['model']

        # Combine statistics where possible (such as acc_avg and acc_std => acc_avg \pm acc_std)
        metrics_add = {}
        for model_name in metrics.keys():
            measures = metrics[model_name].getMeasures()
            metrics_add[model_name] = metrics[model_name].copy()
            for measure in metrics[model_name].getMeasures():
                measure_arr = measure.split('_')
                if len(measure_arr) >= 2 and '_'.join([measure_arr[0], 'avg']) in measures and '_'.join([measure_arr[0], 'std']) in measures:
                    avg = metrics[model_name].getValue('_'.join([measure_arr[0], 'avg']))
                    std = metrics[model_name].getValue('_'.join([measure_arr[0], 'std']))
                    metrics_add[model_name].addValue(measure_arr[0], '{:.4f}\u00B1{:.4f}'.format(avg, std) )
                    metrics_add[model_name].removeValue('_'.join([measure_arr[0], 'avg']))
                    metrics_add[model_name].removeValue('_'.join([measure_arr[0], 'std']))
        metrics = metrics_add


        # Create the objecs to hand over to tabulate
        for model_name in metrics.keys():   
            new_row = [model_name]
            for measure in metrics[model_name].getMeasures():
                if measure not in headers:
                    headers.append(measure)
                new_row.append(metrics[model_name].getValue(measure))
            metrics_list_format.append(new_row)

        print( tabulate(metrics_list_format, headers=headers) )
            

    def setRandomState(self, rand_state):
        if type(rand_state) == int:
            self.random_state = rand_state
            self.helper.setRandomState(rand_state)
            
            for model_name in self.models_dict.keys():
                model = self.models_dict[model_name]
                if model.hasConstructorArg('random_state'):
                    model.setConstructorArg('random_state', rand_state)
        else:
            raise ValueError('random state must be integer: {}'.format(rand_state))

    def getName(self):
        return self.name
    def getModels(self):
        return self.models_dict

    def compareModels_2x5cv(self, models=[], X=None, y=None, a=0.05):
        comparison_rows = []
        headers = ['models', 'model1', 'mean1', 'std1', 'model2', 'mean2', 'std2', 'sig/notsig']

        comparisons_ran = dict()

        for model1 in self.models_dict.keys():
            if not model1 in comparisons_ran:
                comparisons_ran[model1] = []
            for model2 in self.models_dict.keys():
                if not model2 in comparisons_ran:
                    comparisons_ran[model2] = []
                if model1 != model2 and ( model1 not in comparisons_ran[model2] ) and ( model2 not in comparisons_ran[model1] ):
                    row = ['{} & {}'.format(model1, model2)]

                    row.append(model1)
                    cv1 = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=self.random_state)
                    scores1 = cross_val_score(self.models_dict[model1].getBuiltModel(), X, y, scoring='accuracy', cv=cv1)

                    row.append(model2)
                    cv2 = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=self.random_state)
                    scores2 = cross_val_score(self.models_dict[model2].getBuiltModel(), X, y, scoring='accuracy', cv=cv1)
                    
                    meanScore1 = np.mean(scores1)
                    meanScore2 = np.mean(scores2)
                    row.append(f'{np.std(scores1):.5f}')
                    row.append(f'{np.std(scores2):.5f}')

                    if meanScore1 > meanScore2:
                        row.append(f'*{meanScore1:.5f}')
                    else:
                        row.append(f'*{meanScore2:.5f}')

                    t, p = paired_ttest_5x2cv(estimator1=self.models_dict[model1].getBuiltModel(), estimator2=self.models_dict[model2].getBuiltModel(), X=X, y=y, scoring='accuracy')
                    if p <= a:
                        row.append('sig')
                    else:
                        row.append('notsig')
                    comparisons_ran[model1].append(model2)
                    comparisons_ran[model2].append(model1)
                    comparison_rows.append(row)

        print( tabulate(comparison_rows, headers=headers) )

    def compareModels_ttest(self, models=[], X=None, y=None, a=0.05):
        comparison_rows = []
        headers = ['model pairs', 'model1', 'mean1', 'std1', 'model2', 'mean2', 'std2', 't-stat', 'p-val', 'sig/notsig']
        print(*headers)
        models = list(self.models_dict.keys())
        assert len(models) >= 2, 'there must be at least 2 models to run ttest statistical comparison'
        betterClassifier = models[0]
        betterCV = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=self.random_state)
        betterScore = cross_val_score(self.models_dict[betterClassifier].getBuiltModel(), X, y, scoring='accuracy', cv=betterCV)
        betterMeanScore = np.mean(betterScore)
        betterStdDev = np.std(betterScore)
        comparison_rows = []
        for model in models[1:]:
            row = [f'{betterClassifier} v {model}']
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=self.random_state)
            score = cross_val_score(self.models_dict[model].getBuiltModel(), X, y, scoring='accuracy', cv=cv)
            meanScore = np.mean(score)
            stdDev = np.std(score)
            if meanScore > betterMeanScore:
                # this is better classifier
                row.extend([model, f'{meanScore:.5f}*', f'{stdDev:.5f}'])
                row.extend([betterClassifier, f'{betterMeanScore:.5f}', f'{betterStdDev:.5f}'])
                betterClassifier = model
                betterMeanScore = meanScore
                betterStdDev = stdDev
            else:
                row.extend([betterClassifier, f'{betterMeanScore:.5f}*', f'{betterStdDev:.5f}'])
                row.extend([model, f'{meanScore:.5f}', f'{stdDev:.5f}'])

            t, p = paired_ttest_5x2cv(estimator1=self.models_dict[betterClassifier].getBuiltModel(), estimator2=self.models_dict[model].getBuiltModel(), X=X, y=y, scoring='accuracy')
            row.extend([f'{t:.3f}', f'{p:.3f}'])
            if p <= a:
                row.append('sig')
            else:
                row.append('notsig')
            comparison_rows.append(row)
            print(*row)
        print( tabulate(comparison_rows, headers=headers) )
