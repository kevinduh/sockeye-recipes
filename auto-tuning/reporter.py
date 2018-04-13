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
    parser.add_argument("--autotunedir", help="root directory of the auto-tuning scripts")
    parser.add_argument("--n-obj", type=int, choices=range(1,3), 
                        help="number of objective to be optimized,"\
                             " 1 for BLEU score optimization,"\
                             " 2 for BLEU score and validation speed.")
    parser.add_argument("--trg-bleu", help="BLEU scores for each population in one generation")
    parser.add_argument("--trg-time", help="validation time for each population in one generation")

    args = parser.parse_args()
    return args

 
def get_bleu(args):
    """Read scores from bleu.scr"""
    with open(args.scr) as f:
        scores_raw = f.readlines()

    bleu = float(scores_raw[0].split(",")[0].split('=')[-1])

    return bleu

def get_time(args):
    """Get validation time"""
    path1 = os.path.join(args.model_path,"metrics")
    path2 = os.path.join(args.model_path,"multibleu.valid_bpe.result")
    time = os.path.getmtime(path2) - os.path.getmtime(path1)
    return time


def report(args, rst, trg):
    """Write the bleu score for the last iteration of the model into genes.scr"""
    try:
        with open(trg, "r") as f:
            scores = f.readlines()
    except:
        scores = []

    with open(trg, "w") as f:
        scores = scores + [str(args.n_pop).zfill(2) + "\t" + str(rst) + "\n"]
        scores = sorted(scores)
        f.writelines(scores)
        f.flush()

def report_bt(args, bleu, time, trg_bleu, trg_time):
    """Report pareto level based on BLEU score and validation time"""
    # write the validation time of the model into time.trg
    report(args, time, trg_time)
    # write the bleu score for the last iteration of the model into bleu.trg
    report(args, bleu, trg_bleu)
    with open(trg_bleu) as f:
        if len(f.readlines())==args.pop:
            os.system("python " + args.autotunedir + "pareto.py -l " + trg_bleu + " -s " + trg_time + " >" + args.trg)

if __name__ == "__main__":
    args = get_arguments()
    bleu = get_bleu(args)

    if args.n_obj == 1:
        report(args, bleu, args.trg)
    else:
        time = get_time(args)
        report_bt(args, bleu, time, args.trg_bleu, args.trg_time)

    # remove decode.output files except the last one
    # decode_files = list(filter(lambda x: "decode.output" in x, os.listdir(args.model_path)))
    # decode_files.sort()
    # for f in decode_files[:-1]:
    #     os.remove(os.path.join(args.model_path,f))

    # remove generated useless files
#    os.system("rm ~/train.sh.*")