#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import explode, col
import pyspark.sql.functions as F


class ALSModelCustom:
    def __init__(self, user_col='user_index', item_col='track_index',
                 rating_col='count', rank=100, max_iter=8, reg_param=0.01, implicit_prefs=True,
                 alpha=0.1, nonnegative=True, cold_start_strategy='drop'):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.model = ALS(rank=rank,
                         maxIter=max_iter,
                         regParam=reg_param,
                         implicitPrefs=implicit_prefs,
                         alpha=alpha,
                         userCol=user_col,
                         itemCol=item_col,
                         ratingCol=rating_col,
                         nonnegative=nonnegative,
                         coldStartStrategy=cold_start_strategy,
                         numBlocks=10000
                         )

    def dump(self, dump_path):
        self.model.save(dump_path)

    @staticmethod
    def load(path):
        instance = ALSModelCustom()
        instance.model = ALSModel.load(path)
        return instance

    def predict(self, test_data):
        test_data_user_ids = test_data.groupBy(self.user_col).count().select('user_index')

        top_all = (
            test_data.select('user_index', 'track_index', 'count')
            .groupBy('user_index')
            .agg(F.collect_set('track_index').alias('true_ranked_list'))
        )
        collapsed_predict = self.model.recommendForUserSubset(test_data_user_ids, numItems=500).\
            withColumn('rec_exp', explode("recommendations")).\
            select(self.user_col, col(f'rec_exp.{self.item_col}')).\
            groupby(self.user_col).agg(F.collect_list(self.item_col).alias("predicted_set"))
        return collapsed_predict.join(top_all, on=self.user_col).drop(self.user_col)

    def fit(self, training_data):
        self.model = self.model.fit(training_data)
