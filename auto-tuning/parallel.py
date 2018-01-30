import os
import multiprocessing
from logger import *
import argparse

qsub_sh = "qsub -sync y -l 'gpu=1' -q g.q ${rootdir}/auto-tuning/train.sh %s"

def get_arguments():
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument("--pop", default=30, type=int, help="population")
    parser.add_argument("--num-devices", default=1, type=int, 
                        help="the number of computation resources allocated")

    args = parser.parse_args()
    return args

def run_train(n_dev):
    os.system(qsub_sh%(str(n_dev)))


def train_parallel(num_devices):
    pool = multiprocessing.Pool(num_devices)
    pool.map(run_train, [n_dev for n_dev in args.pop])


if __name__ == "__main__":
    args = get_arguments()
    train_parallel(args.num_devices)
