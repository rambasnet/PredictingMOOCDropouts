import numpy as np

class Metric:
    def __init__(self, name, fold=None):
        self.name = name
        self.fold_num = fold
        self.values = {}

    def __str__(self):
        return str({self.name: self.values})

    def __repr__(self):
        return str({self.name: self.values})

    def __eq__(self, other):
        if type(other) == Metric:
            return_value = True
            vals = other.getValues()
            for key in vals.keys():
                if not (key in self.values) or self.values[key] != vals[key]:
                    return_value = False

            return return_value
        else:
            raise ValueError('Equals not supported between Metric and {} classes'.format( type(other) ))

    def copy(self):
        m = Metric(name=self.name, fold=self.fold_num)
        for val in self.values.keys():
            m.addValue(val, self.values[val])
        return m

    def addValue(self, m_type, value):
        if m_type != None and value != None and type(m_type) == str and ( type(value) == float or type(value) == int or type(value) == np.float64 or type(value) == str):
            self.values[m_type] = value
        else:
            raise ValueError('Metric.addValue must have \'m_type\' as string and \'value\' as integer or floating point number instead of type(m_type) => {} and type(value) => {}'.format(type(m_type), type(value)))

    def removeValue(self, m_type):
        if m_type != None and type(m_type) == str:
            if m_type in self.values:
                del self.values[m_type]
        else:
            raise ValueError('Metric.removeValue must have \'m_type\' as string, not {}'.format(type(m_type)))

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

    def getMetricWithMeasure(self, m_type='all'):
        # Return a metric with only the data requested, which may be in list format if there is more than one measurement desired
        if m_type == 'all':
            m_type = list( self.values.keys() )

        if type(m_type) == list:
            new_metric = Metric(self.name, fold=self.fold_num)
            for m in m_type:
                new_metric.addValue(m, self.values[m])

            return new_metric.getValues()
        elif type(m_type) == str:
            new_metric = Metric(self.name, fold=self.fold_num)
            new_metric.addValue(m_type, self.values[m_type])

            return new_metric.getValues()
        else:
            raise ValueError('Metric.getMetricWithMeasure must be given either a string of metric or array of strings of metrics desired.')

class MetricsManager:
    def __init__(self):
        self.metrics_list = []

    def clearMetrics(self):
        self.metrics_list = []
    
    def getMetrics(self, model_name='all', m_type='all'):
        # If they want everything, give them everything
        if model_name == 'all' and m_type == 'all':
            return self.metrics_list
        # If they want a list of models, the conditional in the lambda function changes a little bit
        elif type(model_name) == list:
            # This line is a blast! It searches through all of the metrics the manager knows about, and returns all the metrics that have both the name and metrics the user wants in a list
            return [ m for m in self.metrics_list if (m.getName() in model_name) and (m.containsType(m_type) or m_type == 'all') ]
            """return list(
                filter(
                    None, 
                    map( 
                        lambda m : m.getMetricWithMeasure(m_type) if (m.getName() in model_name) and (m.containsType(m_type) or m_type == 'all') else None, 
                        self.metrics_list
                    )
                )
            )"""
        # Return the data requested as per the terrible line below
        else:
            # This line is a blast! It searches through all of the metrics the manager knows about, and returns all the metrics that have both the name and metrics the user wants in a list
            return self.getMetrics(model_name=[model_name], m_type=m_type)
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
        return_dict = {}

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

        # Go through all of the models and print their data one line at a time
        printed_models = []
        for metric in metrics:
            metric_name = metric.getName()
            
            # If the model hasn't already been printed (this can happen if I have multiple folds for one classifier), then print the data
            if metric_name not in printed_models:
                return_dict[metric_name] = Metric(metric_name)

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
                        return_dict[metric_name].addValue('{}_avg'.format(measure),np.mean(vals))
                        return_dict[metric_name].addValue('{}_std'.format(measure),np.std(vals))
                # Make note of the model we just printed. We don't want any repeats
                printed_models.append(metric_name)
        return return_dict
