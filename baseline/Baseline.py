#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit show_parquet.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass

# Importing packages
import pyspark
import random
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql import Row
from pyspark.ml.feature import  StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql.functions import col,avg,countDistinct,sum, rank
from pyspark.sql.functions import row_number,lit
from pyspark.sql.window import Window
from pyspark.sql.functions import explode
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import lit
from pyspark.sql.functions import expr
from pyspark.sql.functions import monotonically_increasing_id



def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    
    train = spark.read.parquet('hdfs:///user/bm106/pub/MSD/cf_train.parquet')
    train = train.drop("__index_level_0__")
    
    validation = spark.read.parquet('hdfs:///user/bm106/pub/MSD/cf_validation.parquet')
    validation = validation.drop("__index_level_0__")
    
    test = spark.read.parquet('hdfs:///user/bm106/pub/MSD/cf_test.parquet')
    test = test.drop("__index_level_0__")
    
    # show the training set
    print("Show the train set")
    train.show()
    
    
    # map the user and track id to numbers
    # this is done by union all ids from train, validation and test, then assign each id with a int
    unique_user_id = train.select('user_id').distinct().\
        union(validation.select('user_id').distinct()).\
        union(test.select('user_id').distinct())
    
    
    w = Window().orderBy(lit('A'))
    unique_user_id = unique_user_id.withColumn("user_index", row_number().over(w))

    
    unique_track_id = train.select('track_id').distinct().\
        union(validation.select('track_id').distinct()).\
        union(test.select('track_id').distinct())
    
    
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
    
    
    print("This is the train set")
    transformed_train.show()
    
    
    
    ################################### Baseline model: popularity #############################3
    
    # first calculate the average utility
    popular_songs_popularity = transformed_train.groupby("track_index").agg(F.avg(col("count")).alias("avg_count")).\
        sort(col("avg_count").desc()).\
            limit(500).select("track_index").\
        agg(F.collect_list(col("track_index")).alias("popular_songs")).\
        withColumn("id_", monotonically_increasing_id())
    
    
    collapsed_test = transformed_test.groupby("user_index").\
        agg(F.collect_set("track_index").\
        alias("actual_set")).\
        withColumn("id_", monotonically_increasing_id())
    
    # join with the test set
    res_popularity = collapsed_test.join(popular_songs_popularity, "id_", "outer").drop("id_").\
        withColumn("popular_songs", F.last('popular_songs', True).over(w)).\
        select("popular_songs", "actual_set")
    
    print("popularity: outcome")
    res_popularity.show()
    
    
    # change into tuple
    metrics_popularity = RankingMetrics(res_popularity.rdd.map(tuple))
    print("Mean average precision of the base line model is: ",  metrics_popularity.meanAveragePrecision)
    
    
    ################################### Baseline model: factorization #############################3
    beta1 = 5 # damping factor for user
    beta2 = 5 # damping factor for item
    
    
    # first calculate the mu
    mu = transformed_train.agg(avg(col("count"))).collect()
    mu = mu[0][0]
    
    
    # count the number of users
    user_counts = transformed_train.groupBy("user_index").agg(F.count("user_index").alias("count_user_index"))
    track_counts = transformed_train.groupBy("track_index").agg(F.count("track_index").alias("count_track_index"))
    
    
    # sum over the user
    user_sums = transformed_train.groupBy("user_index").agg(sum("count").alias("sum_counts"))
    
    # user factor
    b_u = user_sums.join(user_counts, user_sums.user_index == user_counts.user_index, how='inner').\
        withColumn("mu_*_count",col("count_user_index") * mu).\
        withColumn("b_u",(col("sum_counts") - col("mu_*_count"))/(col("count_user_index") + beta1)).\
        select(user_sums.user_index, col("b_u"))
    
    
    # residual = R - mu - b_u
    track_residual = transformed_train.join(b_u, transformed_train.user_index == b_u.user_index, how='left').\
        withColumn("track_residual", (col("count") - col("b_u") - mu)).\
        select(transformed_train.user_index, transformed_train.track_index,col("b_u"), col("track_residual"))
    
    print("residual")
    track_residual.show()
    
    
    # item factor
    b_i = track_residual.groupBy("track_index").agg(sum("track_residual").alias("sum_residuals")).\
        join(track_counts, track_residual.track_index == track_counts.track_index, how='inner').\
        withColumn("b_i",(col("sum_residuals")/(col("count_track_index") + beta2))).\
        select(track_residual.track_index, col("b_i"))
    
    
    popular_songs_factorization = b_i.sort(col("b_i").desc()).\
        limit(500).select("track_index").\
        agg(F.collect_list(col("track_index")).alias("popular_songs")).\
        withColumn("id_", monotonically_increasing_id())
    
    
    res_factorization = collapsed_test.join(popular_songs_factorization, "id_", "outer").drop("id_").\
        withColumn("popular_songs", F.last('popular_songs', True).over(w)).\
        select("popular_songs", "actual_set")
    
    print("factorization: outcome")
    res_factorization.show()
    
    
    metrics_factorization = RankingMetrics(res_factorization.rdd.map(tuple))
    print("Mean average precision of the base line model is: ",  metrics_factorization.meanAveragePrecision)
    
    
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder \
                        .appName('model') \
                        .config("spark.executor.memory", "8g") \
                        .getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)