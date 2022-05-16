#
# Created by
#

from pyspark.sql import SparkSession


def build_spark_session(name, mem_limit='8g'):
    spark = SparkSession \
        .builder \
        .appName(name) \
        .config('spark.executor.memory', mem_limit) \
        .config('spark.executor.instances', 4) \
        .config('spark.executor.cores', 4) \
        .getOrCreate()
    return spark
