#
# Created by Taotao Tan (tt1922@nyu.edu)
#

import getpass
from recommender_system.utils import build_spark_session
import argparse
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import functions as F
from pyspark.sql.functions import lit
from pyspark.sql.window import Window


def popularity_train(spark, netID, suffix):
    data = spark.read.parquet(f'hdfs:/user/{netID}/data_train_preprocessed_{suffix}.parquet')
    data_test = spark.read.parquet(f'hdfs:/user/{netID}/data_test_preprocessed_{suffix}.parquet')

    # first calculate the average utility
    popular_songs_popularity = data.groupby("track_index").agg(F.avg(col("count")).alias("avg_count")). \
        sort(col("avg_count").desc()). \
        limit(500).select("track_index"). \
        agg(F.collect_list(col("track_index")).alias("popular_songs")). \
        withColumn("id_", monotonically_increasing_id())

    collapsed_test = data_test.groupby("user_index"). \
        agg(F.collect_set("track_index"). \
            alias("actual_set")). \
        withColumn("id_", monotonically_increasing_id())

    # join with the test set
    w = Window().orderBy(lit('A'))
    res_popularity = collapsed_test.join(popular_songs_popularity, "id_", "outer").drop("id_"). \
        withColumn("popular_songs", F.last('popular_songs', True).over(w)). \
        select("popular_songs", "actual_set")

    print("popularity: outcome")
    res_popularity.show()

    # change into tuple
    metrics_popularity = RankingMetrics(res_popularity.rdd.map(tuple))
    print("Mean average precision of the base line model is: ", metrics_popularity.meanAveragePrecision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--volume', nargs='?', default='0.001',
                        help='data volume to run script on, from 0.0 to 1.0')

    args = parser.parse_args()

    suffix = str(args.volume).replace('.', 'p')
    netID = getpass.getuser()
    spark = build_spark_session('popularity-train')
    popularity_train(spark, netID, suffix)
