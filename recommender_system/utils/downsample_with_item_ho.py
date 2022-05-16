#
# Created by Xiao Quan (xq264@nyu.edu)
# Adapted by Maksim Eremeev (mae9785@nyu.edu)
#


from recommender_system.utils import build_spark_session
import argparse


def get_item(item):
    return item['item_index']


def downsample(spark, repartition_param=100, data_volume=0.01):
    file_path_train = 'data_train_preprocessed_1p0.parquet'
    file_path_validation = 'data_validation_preprocessed_1p0.parquet'
    file_path_test = 'data_test_preprocessed_1p0.parquet'

    data_volume_str = str(data_volume).replace('.', 'p')
    training_set = spark.read.parquet(file_path_train).repartition(repartition_param)

    items_ids = training_set.select('item_index').distinct()
    excluded_item_ids = items_ids.sample(False, 0.25, 12345)
    excluded_item_ids_set = set(excluded_item_ids.rdd.map(get_item).collect())

    excluded_item_ids.write.mode('overwrite').parquet(f'excluded_items.parquet')

    training_set = training_set.sample(False, data_volume, 12345)
    training_set_filtered = training_set.filter(training_set['item_index'].isin(excluded_item_ids_set) == False)
    training_set_filtered.write.mode('overwrite').parquet(f'data_train_preprocessed_ho_{data_volume_str}.parquet')

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

    spark = build_spark_session('downsampling-with-items-hold-out')
    downsample(spark, args.repartition, args.volume)
