#
# Created by Taotao Tan (tt1922@nyu.edu)
#

import getpass
from recommender_system.utils import build_spark_session
import argparse
from pyspark.sql.functions import col, avg
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import functions as F
from pyspark.sql.functions import lit
from pyspark.sql.window import Window


def factorization_train(spark, netID, suffix):
    data = spark.read.parquet(f'hdfs:/user/{netID}/data_train_preprocessed_{suffix}.parquet')
    data_test = spark.read.parquet(f'hdfs:/user/{netID}/data_test_preprocessed_{suffix}.parquet')

    beta1 = 5  # damping factor for user
    beta2 = 5  # damping factor for item

    w = Window().orderBy(lit('A'))

    # first calculate the mu
    mu = data.agg(avg(col("count"))).collect()
    mu = mu[0][0]

    # count the number of users
    user_counts = data.groupBy("user_index").agg(F.count("user_index").alias("count_user_index"))
    track_counts = data.groupBy("track_index").agg(F.count("track_index").alias("count_track_index"))

    # sum over the user
    user_sums = data.groupBy("user_index").agg(sum("count").alias("sum_counts"))

    # user factor
    b_u = user_sums.join(user_counts, user_sums.user_index == user_counts.user_index, how='inner'). \
        withColumn("mu_*_count", col("count_user_index") * mu). \
        withColumn("b_u", (col("sum_counts") - col("mu_*_count")) / (col("count_user_index") + beta1)). \
        select(user_sums.user_index, col("b_u"))

    # residual = R - mu - b_u
    track_residual = data.join(b_u, data.user_index == b_u.user_index, how='left'). \
        withColumn("track_residual", (col("count") - col("b_u") - mu)). \
        select(data.user_index, data.track_index, col("b_u"), col("track_residual"))

    print("residual")
    track_residual.show()

    # item factor
    b_i = track_residual.groupBy("track_index").agg(sum("track_residual").alias("sum_residuals")). \
        join(track_counts, track_residual.track_index == track_counts.track_index, how='inner'). \
        withColumn("b_i", (col("sum_residuals") / (col("count_track_index") + beta2))). \
        select(track_residual.track_index, col("b_i"))

    popular_songs_factorization = b_i.sort(col("b_i").desc()). \
        limit(500).select("track_index"). \
        agg(F.collect_list(col("track_index")).alias("popular_songs")). \
        withColumn("id_", monotonically_increasing_id())

    collapsed_test = data_test.groupby("user_index").\
        agg(F.collect_set("track_index").\
        alias("actual_set")).\
        withColumn("id_", monotonically_increasing_id())

    res_factorization = collapsed_test.join(popular_songs_factorization, "id_", "outer").drop("id_"). \
        withColumn("popular_songs", F.last('popular_songs', True).over(w)). \
        select("popular_songs", "actual_set")

    print("factorization: outcome")
    res_factorization.show()

    metrics_factorization = RankingMetrics(res_factorization.rdd.map(tuple))
    print("Mean average precision of the base line model is: ", metrics_factorization.meanAveragePrecision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--volume', nargs='?', default='0.001',
                        help='data volume to run script on, from 0.0 to 1.0')

    args = parser.parse_args()

    suffix = str(args.volume).replace('.', 'p')
    netID = getpass.getuser()
    spark = build_spark_session('factorization-train')
    factorization_train(spark, netID, suffix)
