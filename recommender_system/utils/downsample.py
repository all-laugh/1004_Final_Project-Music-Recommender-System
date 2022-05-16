#
# Created by Xiao Quan (xq264@nyu.edu)
#


from recommender_system.utils import build_spark_session
import argparse


def downsample(spark, repartition_param=100, data_volume=0.01):
    file_path_train = 'data_train_preprocessed_1p0.parquet'
    file_path_validation = 'data_validation_preprocessed_1p0.parquet'
    file_path_test = 'data_test_preprocessed_1p0.parquet'

    data_volume_str = str(data_volume).replace('.', 'p')
    training_set = spark.read.parquet(file_path_train).repartition(repartition_param)
    training_set = training_set.sample(False, data_volume, 12345)
    training_set.write.mode('overwrite').parquet(f'data_train_preprocessed_{data_volume_str}.parquet')

    validation_set = spark.read.parquet(file_path_validation).repartition(repartition_param)
    validation_set = validation_set.sample(False, data_volume, 12345)
    validation_set.write.mode('overwrite').parquet(f'data_validation_preprocessed_{data_volume_str}.parquet')

    test_set = spark.read.parquet(file_path_test).repartition(repartition_param)
    test_set = test_set.sample(False, data_volume, 12345)
    test_set.write.mode('overwrite').parquet(f'data_test_preprocessed_{data_volume_str}.parquet')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--volume', nargs='?', default=0.01, type=float,
                        help='data volume for downsampling, from 0.0 to 1.0')
    parser.add_argument('-r', '--repartition', nargs='?', default=10000, type=int,
                        help='repartition value')
    args = parser.parse_args()

    spark = build_spark_session('downsampling')
    downsample(spark, args.repartition, args.volume)

