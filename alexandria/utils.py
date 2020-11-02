# Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


class Helper:
    def __init__(self):
        self.random_state = 0
        self.initializeDefaultModels()

    def initializeDefaultModels(self):
        self.default_model_objs = {
                'rf:regressor': [
                        RandomForestRegressor,
                        {'random_state':self.random_state}],
                'rf:classifier': [
                        RandomForestClassifier,
                        {'random_state':self.random_state}],
                'dt:regressor': [
                        DecisionTreeRegressor,
                        {'random_state':self.random_state}],
                'dt:classifier': [
                        DecisionTreeClassifier,
                        {'random_state':self.random_state}],
                'ab:classifier': [
                        AdaBoostClassifier,
                        {'random_state':self.random_state}],
                'ab:regressor': [
                        AdaBoostRegressor,
                        {'random_state':self.random_state}],
                'knn:classifier': [
                        KNeighborsClassifier,
                        {}],
                'knn:regressor': [
                        KNeighborsRegressor,
                        {}],
                'lda:classifier': [
                        LinearDiscriminantAnalysis,
                        {}],
                'lr:classifier': [
                        LogisticRegression,
                        {'random_state':self.random_state}],
                'lr:regressor': [LogisticRegression, {'random_state':self.random_state}],
                'nb:classifier': [
                        GaussianNB,
                        {}],
                'svm:classifier': [
                        SVC,
                        {'random_state':self.random_state}],
                'mlp:classifier': [
                        MLPClassifier,
                        {'random_state':self.random_state}],
                'mlp:regressor': [
                        MLPRegressor,
                        {'random_state':self.random_state}],
                'xgb:classifier': [XGBClassifier, {'random_state': self.random_state}],
                'xgb:regressor': [XGBRegressor, {'random_state': self.random_state}]
                
        }

    def getDefaultModel(self, model):
        if type(model) == str:
            model = model.split(':')
            for i in range(len(model)):
                model_name = ':'.join( model[:i+1] )
                if model_name in self.default_model_objs:
                    return self.default_model_objs[model_name][0]
        else:
            raise ValueError('No default model found for {}'.format(str(model)))

    def getDefaultArgs(self, model):
        if type(model) == str and model in self.default_model_objs:
            return self.default_model_objs[model][1]
        else:
            raise ValueError('No default model found for {}'.format(str(model)))

    def setDefaultArgs(self, model, args):
        if type(model) == str and model in self.default_model_objs:
            if type(args) == dict:
                if len(self.default_model_objs[model]) < 2:
                    raise ValueError('Illegal state! default_model_objs[{}] cannot have fewer than 2 items!'.format(model))
                self.default_model_objs[model][1] = args
            else:
                raise ValueError('Arguments must be in dictionary format, not {} type!'.format( str(type(args)) ))
        else:
            raise ValueError('Default model cannot be found: {}'.format(str(model)))

    def setRandomState(self, rand_state):
        if type(rand_state) == int:
            self.random_state = rand_state
            self.initializeDefaultModels()
        else:
            raise ValueError('Random state must be an integer: {}'.format(str(rand_state)))

    def getBuiltModel(self, model_name):
        model = self.getDefaultModel(model_name)
        args = self.getDefaultArgs(model_name)
        return model(**args)