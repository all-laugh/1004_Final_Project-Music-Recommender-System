#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import getpass
from recommender_system.utils import build_spark_session
from recommender_system.models import ALSModelCustom
import argparse
from pyspark.mllib.evaluation import RankingMetrics


def als_train(spark, netID, suffix, model_name, use_validation, rank, reg_param):
    data = spark.read.parquet(f'hdfs:/user/{netID}/data_train_preprocessed_{suffix}.parquet')

    model = ALSModelCustom(rank=rank, reg_param=reg_param)
    model.fit(data)
    
    data_type = 'test'
    if use_validation:
        data_type = 'validation'
    data_test = spark.read.parquet(f'hdfs:/user/{netID}/data_{data_type}_preprocessed_{suffix}.parquet')
    result = model.predict(data_test)

    metrics = RankingMetrics(result.rdd)
    print(
        'map: ', metrics.meanAveragePrecision,
        'ngcd@10: ', metrics.ndcgAt(10),
        'ngcd@50: ', metrics.ndcgAt(50),
        'ngcd@100: ', metrics.ndcgAt(100),
        'ngcd@500: ', metrics.ndcgAt(500),
    )
    model.dump(f'hdfs:/user/{netID}/{model_name}.als_model')


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
    spark = build_spark_session('als-train')
    als_train(spark, netID, suffix, args.model, args.test, args.rank, args.reg)
