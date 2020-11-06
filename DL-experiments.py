# Data management
import os
#from google.colab import drive
import pandas as pd

# Additional Scikit-Learn imports
from sklearn.model_selection import StratifiedKFold

# Scikit-Learn's ML classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

# Additional Metrics
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

# fast.ai v1
# from fastai.tabular import *
# Fast.ai v2 DNN Classifer
from fastai.tabular.all import *

# Keras DNN Classifier
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout
from keras.regularizers import l2
from keras.utils import to_categorical, normalize
from keras import backend as K


# Class used to help manage the classifier performance metrics
class Metric:
    def __init__(self, name, fold):
        self.name = name
        self.fold_num = fold
        self.values = {}

    def __str__(self):
        return str({self.name: self.values})

    def __repr__(self):
        return str({self.name: self.values})

    def addValue(self, m_type, value):
        if m_type != None and value != None:
            self.values[m_type] = value

    def getValue(self, m_type):
        if m_type in self.values:
            return self.values[m_type]

    def getName(self):
        return self.name

    def getMeasures(self):
        # Retuns all the types of measurements (accuracy or time or whatever you have)
        return self.values.keys()

    def getValues(self):
        return self.values

    def containsType(self, m_type):
        # Checks to see if the measurement type (accuracy for example) is contained in here
        if type(m_type) == list:
            for m in m_type:
                if m not in self.values:
                    return False
            return True
        elif type(m_type) == str:
            if m_type in self.values:
                return True
            else:
                return False
        else:
            return False

    def getMetricWithMeasure(self, m_type):
        # Return a metric with only the data requested, which may be in list format if there is more than one measurement desired
        if type(m_type) == list:
            new_metric = Metric(self.name, fold=self.fold_num)
            for m in m_type:
                new_metric.addValue(m, self.values[m])

            return new_metric
        elif type(m_type) == str:
            new_metric = Metric(self.name, fold=self.fold_num)
            new_metric.addValue(m_type, self.values[m_type])

            return new_metric

class MetricsManager:
    def __init__(self):
        self.metrics_list = []
    
    def getMetrics(self, model_name='all', m_type='all'):
        # If they want everything, give them everything
        if model_name == 'all' and m_type == 'all':
            return self.metrics_list
        # If they want a list of models, the conditional in the lambda function changes a little bit
        elif type(model_name) == list:
            # This line is a blast! It searches through all of the metrics the manager knows about, and returns all the metrics that have both the name and metrics the user wants in a list
            return list(
                filter(
                    None, 
                    map( 
                        lambda m : m.getMetricWithMeasure(m_type) if (m.getName() in model_name) and (m.containsType(m_type) or m_type == 'all') else None, 
                        self.metrics_list
                    )
                )
            )
        # Return the data requested as per the terrible line below
        else:
            # This line is a blast! It searches through all of the metrics the manager knows about, and returns all the metrics that have both the name and metrics the user wants in a list
            return list(
                filter(
                    None, 
                    map( 
                        lambda m : m.getMetricWithMeasure(m_type) if (m.getName() == model_name or model_name == 'all') and (m.containsType(m_type) or m_type == 'all') else None, 
                        self.metrics_list
                    )
                )
            )

    def addMetric(self, metric):
        self.metrics_list.append(metric)

    def printMeasures(self, model='all', metrics='all'):
        # Acquire all of the metrics the user wants us to print first so there's no weird filtering going on later
        metrics = self.getMetrics(model_name=model, m_type=metrics)

        # Figure out all of the metrics that are going to be available and figure out their ordering
        #   If we are printing time and accuracy data, we want the columns to be consistent
        measurements = []
        for metric in metrics:
            metric_measures = metric.getMeasures()
            for measure in metric_measures:
                if measure not in measurements:
                    measurements.append(measure)

        # Formatting for the header, we need to print the model column name, then each of the values collected
        print('{:10}'.format('model'), end='')
        for measure in measurements:
            print('{:11}'.format(measure), end='')
        print('\n', end='')
        print('-------'*(len(measurements)+1))

        # Go through all of the models and print their data one line at a time
        printed_models = []
        for metric in metrics:
            metric_name = metric.getName()
            
            # If the model hasn't already been printed (this can happen if I have multiple folds for one classifier), then print the data
            if metric_name not in printed_models:
                print('{:9}'.format(metric_name), end='')

                # Get all of the values stored in this metric (it's in a dictionary)
                metric_values = metric.getValues()
                
                # Go through all of the measurement values in the order as determined above
                for measure in measurements:
                    if measure in metric_values:
                        # We need to go through all of the data and calculate the mean and std deviation from each fold
                        #  If there are no folds, then this won't cause any damage (Keep unique identifiers!)
                        vals = []
                        for m in metrics:
                            if m.getName() == metric_name:
                                vals.append(m.getValues()[measure])
                        # Print the calcuated mean and standard deviations
                        print('{:6.2f}\u00B1{:6<.2f}'.format(np.mean(vals), np.std(vals)), end='')
                    # If there is no value to print, just skip over this element
                    else:
                        print(' '*11, end='')
                # Make note of the model we just printed. We don't want any repeats
                printed_models.append(metric_name)
                print('\n', end='')

def model_eval(model, model_name, fold, X_train, X_test, Y_train, Y_test, metrics_manager):
      print(f'Training and evaulating model: {model_name}')
      model.fit(X_train, Y_train)

      y_pred = model.predict(X_test)
      acc = accuracy_score(Y_test, y_pred)
      #rec = recall_score(Y_test, y_pred, average='weighted')
      #prec = precision_score(Y_test, y_pred, average='weighted')
      #auc = roc_auc_score(Y_test, y_pred)
      #f1 = f1_score(Y_test, y_pred)

      m = Metric(model_name, fold=fold)
      m.addValue('acc', 100*acc)
      #m.addValue('rec', 100*rec)
      #m.addValue('prec', 100*prec)
      #m.addValue('auc', 100*auc)
      #m.addValue('f1', 100*f1)
      
      metrics_manager.addMetric(m)


def train_and_eval_ML(X_train, X_test, y_train, y_test, metrics_manager, fold, quick_test=False):
    """ Train and evulate Traditional ML classifiers from sci-kit learn

        Description: This function will train all the models on the given feature set of the X (data) for predicting y (target) and add the acquired metrics 
        to the MetricsManager object from the user

        Args: 
            X => pd.DataFrame object containing the data
            y => pd.Series object containings the target classifications
            feature_set => list of features in X to use for training
            metrics_manager => MetricsManager object (custom)

        Returns:
            None
        
        Classifer names used as keys for the manager:
                        XGBoost Classifier => xgb
                        Random Forest => rf
                        Decision Tree => dt
                        k-Nearest Neighbors => knn
                        Support Vector Machine => svm
                        Logistic Regression => lr
                        Linear Discriminant Analysis => lda
                        AdaBoost => ab
                        Naive Bayes => nb

    """
    random_state = 100
    if quick_test:
        # Random Forest Model
        rf = RandomForestClassifier(random_state=random_state)
        model_eval(rf, 'rf', fold, X_train, X_test, y_train, y_test, metrics_manager)

        # XGBoost Classifier
        xgb = XGBClassifier()
        model_eval(xgb, 'xgb', fold, X_train, X_test, y_train, y_test, metrics_manager)
        return

    # Random Forest Model
    rf = RandomForestClassifier(random_state=random_state)
    model_eval(rf, 'rf', fold, X_train, X_test, y_train, y_test, metrics_manager)

    # XGBoost Classifier
    xgb = XGBClassifier()
    model_eval(xgb, 'xgb', fold, X_train, X_test, y_train, y_test, metrics_manager)

    # AdaBoost Model
    ab = AdaBoostClassifier(random_state=random_state)
    model_eval(ab, 'ab', fold, X_train, X_test, y_train, y_test, metrics_manager)
    
    # Decision Tree Model
    dt = DecisionTreeClassifier(random_state=random_state)
    model_eval(dt, 'dt', fold, X_train, X_test, y_train, y_test, metrics_manager)

    # k-Nearest Neighbors Model
    knn = KNeighborsClassifier()
    model_eval(knn, 'knn', fold, X_train, X_test, y_train, y_test, metrics_manager)

    # Support Vector Machine Model
    svm = SVC(random_state=random_state)
    model_eval(svm, 'svm', fold, X_train, X_test, y_train, y_test, metrics_manager)

    # Logistic Regression Model
    lr = LogisticRegression(random_state=random_state)
    model_eval(lr, 'lr', fold, X_train, X_test, y_train, y_test, metrics_manager)

    # Linear Discriminant Analysis Model
    lda = LinearDiscriminantAnalysis()
    model_eval(lda, 'lda', fold, X_train, X_test, y_train, y_test, metrics_manager)

    # Naive Bayes Model
    nb = GaussianNB()
    model_eval(nb, 'nb', fold, X_train, X_test, y_train, y_test, metrics_manager)

def train_and_eval_DNN(df, X_train, X_test, y_train, y_test, y_names, feature_set, metrics_manager, fold):
    """ Train and Evaulate Deep Neural Networks

    Args:
    df => pandas dataframe
    fold => n-fold cross validation
    
    Classifier names used as key in metrics_manager
        Keras-TensorFlow => keras
        Fast.ai => fastai
    Returns:
    None
    """

    # Keras-TensorFlow DNN Model
    print('Training and Evaluating Keras-Tensoflow...')
    dnn_keras = Sequential(layers=[
                                Dense(128, kernel_regularizer=l2(0.001), activation='relu',input_shape=(len(X_train.columns),)),
                                BatchNormalization(),
                                Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
                                BatchNormalization(),
                                Dense(y_train.nunique(), activation='softmax')
    ])
    dnn_keras.compile(
        optimizer='adam', 
        loss='binary_crossentropy')
    
    dnn_keras.fit(X_train, pd.get_dummies(y_train), epochs=100, verbose=0, batch_size=512)
    #loss, acc = dnn_keras.evaluate(X_test, pd.get_dummies(y_test), verbose=0)
    y_pred = dnn_keras.predict_classes(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    m = Metric('Keras-Tensorflow', fold=fold)
    m.addValue('acc', 100*acc)
    m.addValue('bal-acc', 100*bal_acc)
    m.addValue('rec', 100*rec)
    m.addValue('prec', 100*prec)
    m.addValue('auc', 100*auc)
    m.addValue('f1', 100*f1)
    metrics_manager.addMetric(m)
    metrics_manager.printMeasures()

    # Fast.ai DNN Model, v.2
    print('Training and Evaluating Fast.ai...')
    splits = RandomSplitter(valid_pct=0.2)(range_of(X_train))
    #print(feature_set)
    #print(df[:5])
    tp = TabularPandas(df, procs=[],
                        cat_names= [],
                        cont_names = list(feature_set),
                        y_names= y_names,
                        splits=splits)

    dls = tp.dataloaders(bs=64)
    #dls.show_batch()
    #return
    dnn_fastai = tabular_learner(dls, metrics=accuracy)
    dnn_fastai.fit_one_cycle(5)

    # acquire predictions
    y_pred = []
    #print('Length of test set: {}'.format(len(y_test)))
    for j in range(len(y_test)):
        row, clas, probs = dnn_fastai.predict(X_test.iloc[j])
        #print(clas)
        pred = 0
        if clas >= tensor(0.5):
            pred = 1
        y_pred.append(pred)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    m = Metric('fastai', fold=fold)
    m.addValue('acc', 100*acc)
    m.addValue('bal-acc', 100*bal_acc)
    m.addValue('rec', 100*rec)
    m.addValue('prec', 100*prec)
    m.addValue('auc', 100*auc)
    m.addValue('f1', 100*f1)
    metrics_manager.addMetric(m)



def train_and_eval(df, X, y, feature_set, y_names, metrics_manager, fold=5, quick_test=False):
    """Train and Eval wrapper function

    Args:
    df => pandas dataframe
    X => dataset
    y => truth
    featur_set => set of feature names

    Returns:
    None
    """

    if quick_test:
        fold = 2

    X = X[feature_set]

    print(f'Running {fold}-fold cross validation evaluation')
    print(f'Training with {len(X.columns)} features')

    random_state=100

    # Create stratified, n-fold cross validation object
    sss = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)

    i=1

    # Experiment with n-fold cross validation

    for train_idx, test_idx in sss.split(X, y):

        print(f'fold num {i}')

        # Split the data into the training and testing sets
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        #train_and_eval_ML(X_train, X_test, y_train, y_test, metrics_manager, i, quick_test)
        # train_and_eval_DNN(df, X_train, X_test, y_train, y_test, y_names, feature_set, metrics_manager, fold)
        train_and_eval_DNN(df, X_train, X_test, y_train, y_test, y_names, feature_set, metrics_manager, i)

        i += 1


def KDDExperiments():
    print('DL Experiments on KDD-cup15 Dataset')
    path = 'data/kddcup15'
    db_path = os.path.join(path, 'kdd_all_normalized_features.csv')
    df = pd.read_csv(db_path)
    # Features that could lead to overfitting the models
    bad_features = ['enrollment_id']
    df.drop(labels=bad_features, axis='columns', inplace=True)
    dep_var = 'truth'
    X = df.loc[:, df.columns != dep_var]
    y = df[dep_var]
    mm = MetricsManager()
    fold = 5
    quick_test = False
    #y_names = ['truth']
    # train_and_eval(df, X, y, feature_set, y_names, metrics_manager, fold=5, quick_test=False)
    train_and_eval(df, X, y, X.columns, dep_var, mm, fold, quick_test)
    mm.printMeasures()

def XeutangxExperiments():
    path = 'data/xeutangx'
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
    mm = MetricsManager()
    fold = 5
    quick_test = False
    #y_names = ['truth']
    # train_and_eval(df, X, y, feature_set, y_names, metrics_manager, fold=5, quick_test=False)
    train_and_eval(df, X, y, X.columns, dep_var, mm, fold, quick_test)
    mm.printMeasures()

def KDDStanford():
    print('DL Experiments Stanford KDD-cup15 Dataset')
    path = 'data/stanford'
    db_path = os.path.join(path, 'normalized_all_features.csv')
    df = pd.read_csv(db_path, index_col=0)
    # Features that could lead to overfitting the models
    bad_features = ['enrollment_id']
    df.drop(labels=bad_features, axis='columns', inplace=True)
    dep_var = 'label'
    X = df.loc[:, df.columns != dep_var]
    y = df[dep_var]

    mm = MetricsManager()
    fold = 5
    quick_test = False
    #y_names = ['truth']
    # train_and_eval(df, X, y, feature_set, y_names, metrics_manager, fold=5, quick_test=False)
    train_and_eval(df, X, y, X.columns, dep_var, mm, fold, quick_test)
    mm.printMeasures()
    

if __name__ == "__main__":
    #KDDExperiments()
    #KDDStanford()
    XeutangxExperiments()