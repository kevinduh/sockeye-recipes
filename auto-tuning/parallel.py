import os
import multiprocessing
from logger import *
import argparse

qsub_sh = "qsub -sync y -l 'gpu=1' -q g.q ${rootdir}/auto-tuning/train.sh %s %s %s %s %s"

def get_arguments():
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument("--hyperparams", type=str, help="the hyperparams file")
    parser.add_argument("--pop", default=30, type=int, help="population")
    parser.add_argument("--num-devices", default=1, type=int, 
                        help="the number of computation resources allocated")
    parser.add_argument("--generation_path", type=str, help="path to current generation folder")
    parser.add_argument("--gene",type=str,help="path to gene files")
    parser.add_argument("--n-generation", type=int, help="current generation")

    args = parser.parse_args()
    return args

def run_train(hyparam, n_dev, gen_path, gene, n_gen):
    logging.info("Start training model %s ......"%(str(n_dev)))
    os.system(qsub_sh%(hyparam, str(n_dev), gen_path, gene, str(n_gen)))
    logging.info("Finish training model %s ......"%(str(n_dev)))

def train_parallel(num_devices):
    pool = multiprocessing.Pool(num_devices)
    pool.map(run_train, [n_dev for n_dev in range(args.pop)])


if __name__ == "__main__":
    args = get_arguments()
    train_parallel(args.hyperparams, args.num_devices, args.generation_path, args.gene, args.n_generation)
