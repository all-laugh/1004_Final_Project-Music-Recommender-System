#
# Created by Maksim Eremeev (mae9785@nyu.edu)
# Be aware: this works slowly on spark < 3.0!
#

import getpass
from recommender_system.utils import build_spark_session
from recommender_system.models import ALSModelCustom
from recommender_system.models import LinearRegressionMatrix
import argparse
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.linalg import Vectors


def get_feature(item):
    return item['id'], item['features']


def als_cs_train(spark, netID, suffix, model_name, use_validation, rank, reg_param):
    data = spark.read.parquet(f'hdfs:/user/{netID}/data_train_preprocessed_ho_{suffix}.parquet')

    model = ALSModelCustom(rank=rank, reg_param=reg_param)
    model.fit(data)

    model.dump(f'hdfs:/user/{netID}/{model_name}.als_model')

    items_features = spark.read.parquet(f'hdfs:/user/{netID}/{model_name}.als_model/itemFactors')
    items_base_features = spark.read.parquet(f'hdfs:/user/{netID}/item_timbre_features.parquet')

    items_features_list = items_features.rdd.map(get_feature).collect()
    items_base_features_list = items_base_features.rdd.map(get_feature).collect()

    items_features_dict = {}
    for item in items_features_list:
        items_features_dict[item[0]] = item[1]

    feature_set = []
    for i in range(100):
        current_feature = []
        for current_item_base_features in items_base_features_list:
            current_item_features = items_features_dict[current_item_base_features[0]]
            current_feature += [(current_item_features[1][i], Vectors.dense(*current_item_base_features[1]))]
        feature_set += [spark.createDataFrame(current_feature, ["label", "features"])]

    regression = LinearRegressionMatrix(100)
    regression.fit(feature_set)

    excluded_items = spark.read.parquet(f'hdfs:/user/{netID}/excluded_items.parquet')
    excluded_items_list = excluded_items.rdd.map(get_feature).collect()

    feature_set, ids = [], []
    for i, item in enumerate(excluded_items_list):
        base_features = items_base_features[item]
        ids[i] = item
        feature_set += [Vectors.dense(*base_features)]

    prediction = regression.predict(feature_set)

    to_append = []
    for i in range(len(prediction)):
        latent_repr = []
        for j in range(100):
            latent_repr += [prediction[j][i]]
        to_append += [(ids[i], latent_repr)]

    to_append_df = spark.createDataFrame(to_append, ["id", "features"])
    to_append_df.write.mode('append').parquet(f'hdfs:/user/{netID}/{model_name}.als_model/itemFactors')

    model_new = ALSModelCustom.load(f'hdfs:/user/{netID}/{model_name}.als_model')

    data_type = 'test'
    if use_validation:
        data_type = 'validation'
    data_test = spark.read.parquet(f'hdfs:/user/{netID}/data_{data_type}_preprocessed_{suffix}.parquet')
    result = model_new.predict(data_test)

    metrics = RankingMetrics(result.rdd)
    print(
        'map: ', metrics.meanAveragePrecision,
        'ngcd@10: ', metrics.ndcgAt(10),
        'ngcd@50: ', metrics.ndcgAt(50),
        'ngcd@100: ', metrics.ndcgAt(100),
        'ngcd@500: ', metrics.ndcgAt(500),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', nargs='?', default='als_base',
                        help='unique name of the trained ALS model to dump')

    parser.add_argument('-v', '--volume', nargs='?', default='0.001',
                        help='data volume to run script on, from 0.0 to 1.0')

    parser.add_argument('-t', '--test', nargs='?', default=False,
                        help='use test data for inference')

    parser.add_argument('-r', '--rank', nargs='?', default=10,
                        help='rank parameter for the model')

    parser.add_argument('-p', '--reg', nargs='?', default=0.1,
                        help='regularization parameter parameter for the model')

    args = parser.parse_args()

    suffix = str(args.volume).replace('.', 'p')
    netID = getpass.getuser()
    spark = build_spark_session('als-cs-train-evaluate')
    als_cs_train(spark, netID, suffix, args.model, args.test, args.rank, args.reg)
