#
# Created by Xiao Quan (xq264@nyu.edu)
#


import os
import glob
import numpy as np
import h5py

msd_path = '/scratch/work/courses/DSGA1004-2021/MSD/data'
hf = h5py.File('/scratch/xq264/MSD_track_id_timbre.h5', 'w')

assert os.path.isdir(msd_path), 'wrong path'  # sanity check
import hdf5_getters as GETTERS


def apply_to_all_files(basedir, func=lambda x: x, ext='.h5'):
    i = 0
    # iterate over all files in all subdirectories
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*' + ext))
        # apply function to all files
        for f in files:
            func(f)
            if i % 10000 == 0:
                print("-------{} percent done!-------".format(100 * i / 1000000))
            i += 1
    return i


def func_to_get_ids_and_timbre(filename):
    h5 = GETTERS.open_h5_file_read(filename)

    mfcc = GETTERS.get_segments_timbre(h5)
    mfcc_compressed = np.mean(mfcc, axis=0)

    track_id = GETTERS.get_track_id(h5)
    track_id = str(track_id)[2:-1]

    hf.create_dataset(track_id, data=mfcc_compressed)
    h5.close()


if __name__ == "__main__":
    apply_to_all_files(msd_path, func=func_to_get_ids_and_timbre)
    hf.close()
