#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from pyspark.ml.regression import LinearRegression


class LinearRegressionMatrix:
    def __init__(self, n_regressions, max_iter=10, reg_param=0, elasticnet_param=0.1):
        self.models = []
        for i in range(n_regressions):
            self.models += [LinearRegression(maxIter=max_iter,
                                             regParam=reg_param,
                                             elasticNetParam=elasticnet_param)]

    def fit(self, train_data):
        for i, model in enumerate(self.models):
            self.models[i] = model.fit(train_data[i])

    # We have about 500K unique items so we don't care much about the efficiency here
    def predict(self, test_data):
        result = []
        for item in test_data:
            result += [[]]
            for model in self.models:
                result[-1] += [model.predict(item)]
        return result
