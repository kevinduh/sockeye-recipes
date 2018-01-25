
import numpy as np
import os
from logger import *
import argparse

# Script to generate fake nmt score


def get_arguments():
    parser = argparse.ArgumentParser(description=None)

    # parser.add_argument("--model-desc", default="",
    #                     help="path to model description files's folder")
    parser.add_argument("--trg", default="",
                        help="target file for output score file")

    parser.add_argument("--n-gen", type=int, help="current generation index")
    parser.add_argument("--min-num-epochs", type=int)
    # parser.add_argument("--n-model", type=int,
    #                     help="current model description file index")

    args = parser.parse_args()
    return args



template = "%s BLEU = %.5f, %.5f/%.5f/%.5f/%.5f (BP=0, ratio=0, hyp_len=0, ref_len=0)\n"

if __name__ == "__main__":
    args = get_arguments()
    print(str(args.min_num_epochs))
    n_data = args.min_num_epochs
    scores = np.random.rand(n_data, 5)
    # logging.info("loading file: %s" % (args.model_desc % args.n_model))

    # cur_path = path % (str(n_gen).zfill(2), str(n_model).zfill(2))
    # os.makedirs(cur_path)
    with open(args.trg, "w+") as f:
        for idx, score in enumerate(scores):
            f.writelines(template % (str(idx).zfill(4), *score))
        f.flush()
