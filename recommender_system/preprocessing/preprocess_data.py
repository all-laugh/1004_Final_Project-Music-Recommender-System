#
# Created by Taotao Tan (tt1922@nyu.edu)
#

import getpass
import argparse

from pyspark.sql.functions import col
from pyspark.sql.functions import row_number,lit
from pyspark.sql.window import Window
from recommender_system.utils import build_spark_session


file_path_train = 'hdfs:/user/bm106/pub/MSD/cf_train.parquet'
file_path_validation = 'hdfs:/user/bm106/pub/MSD/cf_validation.parquet'
file_path_test = 'hdfs:/user/bm106/pub/MSD/cf_test.parquet'


def preprocess_data(spark, netID, suffix):
    train = spark.read.parquet(file_path_train)
    train = train.drop("__index_level_0__")

    validation = spark.read.parquet(file_path_validation)
    validation = validation.drop("__index_level_0__")

    test = spark.read.parquet(file_path_test)
    test = test.drop("__index_level_0__")

    unique_user_id = train.select('user_id').distinct() \
                          .union(validation.select('user_id').distinct()).distinct() \
                          .union(test.select('user_id').distinct()).distinct()
   
    w = Window().orderBy(lit('A'))
    unique_user_id = unique_user_id.withColumn("user_index", row_number().over(w))
    
    # do the same thing for the track
    unique_track_id = train.select('track_id').distinct() \
                           .union(validation.select('track_id').distinct()).distinct() \
                           .union(test.select('track_id').distinct()).distinct()
    
    v = Window().orderBy(lit('A'))
    unique_track_id = unique_track_id.withColumn("track_index", row_number().over(v))
    
    # join three tables to substitute string id to int id
    transformed_train = train.join(unique_track_id, unique_track_id.track_id == train.track_id, how='inner').\
                    join(unique_user_id, unique_user_id.user_id == train.user_id, how='inner').\
                    select(col("user_index"), col("track_index"), col("count"))
    
    transformed_validation = validation.join(unique_track_id, unique_track_id.track_id == validation.track_id, how='inner').\
                    join(unique_user_id, unique_user_id.user_id == validation.user_id, how='inner').\
                    select(col("user_index"), col("track_index"), col("count"))
    
    transformed_test = test.join(unique_track_id, unique_track_id.track_id == test.track_id, how='inner').\
                    join(unique_user_id, unique_user_id.user_id == test.user_id, how='inner').\
                    select(col("user_index"), col("track_index"), col("count"))

    transformed_train.write.parquet(f'hdfs:/user/{netID}/data_train_preprocessed_{suffix}.parquet')
    transformed_validation.write.parquet(f'hdfs:/user/{netID}/data_validation_preprocessed_{suffix}.parquet')
    transformed_test.write.parquet(f'hdfs:/user/{netID}/data_test_preprocessed_{suffix}.parquet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--volume', nargs='?', default='0.001', help='data volume to run script on, from 0.0 to 1.0')
    args = parser.parse_args()

    suffix = str(args.volume).replace('.', 'p')
    netID = getpass.getuser()
    spark = build_spark_session('data-preprocessing')
    preprocess_data(spark, netID, suffix)
