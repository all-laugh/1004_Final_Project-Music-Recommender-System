#
# Created by
#

def show_parquet(spark, path):
    parquet_file = spark.read.parquet(path)
    parquet_file.createOrReplaceTempView('fileView')
    parquet_file.show()
