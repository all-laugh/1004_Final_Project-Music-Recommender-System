#
# Created by Maksim Eremeev (mae9785@nyu.edu)
# This runs locally, not via spark-submit. Need to use GPU nodes for that. Then upload parquet file manually to hdfs
#

import getpass
from recommender_system.utils import build_spark_session
from pyspark.ml.linalg import Vectors
import h5py

hf = h5py.File('/scratch/xq264/MSD_track_id_timbre.h5', 'r')


def write_to_parquet(spark, netID):
    all_features = []
    for track_id in hf:
        all_features += [(track_id, Vectors.dense(*hf[track_id]))]
    spark.createDataFrame(all_features, ['ids', 'features'])
    spark.write.mode('overwrite').parquet(f'/scratch/{netID}/items_timbre_features.parquet')


if __name__ == "__main__":
    netID = getpass.getuser()
    spark = build_spark_session('convert-to-parquet')
    write_to_parquet(spark, netID)
    hf.close()
