import os
import numpy as np
from logger import *
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument("--trg", default="",
                        help="target file for evaluation score of each population")
    parser.add_argument("--scr", default="", help="score file")

    parser.add_argument("--pop", default=30, type=int, help="population")
    parser.add_argument("--n-pop", type=int, help="current population index")
    parser.add_argument("--n-gen", type=int, help="current generation index")
    parser.add_argument("--model-path", type=str, help="path to current model")

    args = parser.parse_args()
    return args

# read scores from bleu.scr
def read_scr_file(args):
    with open(args.scr) as f:
        scores_raw = f.readlines()

    # scores = # of iteration * {'avg-sec-per-sent-val', 'bleu-val', 'chrf-val', 'perplexity-train', 'perplexity-val', 'time-elapsed', 'used-gpu-memory'}
    scores = list(map(lambda x: x.strip().replace('=','\t').split('\t')[1:], scores_raw))
    scores = list(map(lambda x: {x[i]:float(x[i+1]) for i in range(0, len(x), 2)}, scores))

    logging.info("File: %s, # of evaluation records: %s" %
                 (args.scr, len(scores)))
    return np.array(scores)

# write the bleu score for the last iteration of the model into genes.scr
def report(args, rst):
    try:
        with open(args.trg, "r") as f:
            scores = f.readlines()
    except:
        logging.info("Creating new file: " + args.trg)
        # os.system("touch " + args.trg)
        scores = []

    with open(args.trg, "w") as f:
        scores = scores + [str(args.n_pop).zfill(2) + "\t" + str(rst) + "\n"]
        logging.info("collecting: %s" % scores[-1][:-1])
        logging.info("# of collected score: %s" % len(scores))
        scores = sorted(scores)
        f.writelines(scores)
        f.flush()

    if len(scores) == args.pop:
        logging.info(
            "Collected evulation result of all population. start next generation...")



if __name__ == "__main__":
    args = get_arguments()
    scores = read_scr_file(args)
    # report perplexity-val
    # report(args, scores[-1]['perplexity-val'])

    # report bleu-val
    report(args, scores[-1]['bleu-val'])

    # remove decode.output files except the last one
    decode_files = list(filter(lambda x: "decode.output" in x, os.listdir(args.model_path)))
    decode_files.sort()
    for f in decode_files[:-1]:
        os.remove(os.path.join(args.model_path,f))

    # remove generated useless files
    os.system("rm ~/train.sh.*")