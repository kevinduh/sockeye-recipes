import cma
import numpy as np
import pickle
import os
import argparse
from logger import *
from collections import OrderedDict

funcs = {'log': lambda x: np.log(x),
         'exp': lambda x: np.exp(x),
         'identity': lambda x: x}
inverse_funcs = {'log': lambda x: (np.around(np.exp(np.abs(x))/2)*2).astype(np.int),
                 'exp': lambda x: np.around(np.log(np.abs(x)), decimals=10),
                 'identity': lambda x: np.abs(x)}

def get_arguments():
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument("--checkpoint", default="checkpoints/es_G%s.pkl",
                        help="path to checkpoint files")
    parser.add_argument("--gene", default="generation_%s/genes/%s.gene",
                        help="path to backup es files")
    parser.add_argument("--params", type=str,
                        help="hyper-parameters to be tuned with initial value")
    parser.add_argument("--map-func", type=str,
                        help="map hyper-parameters into same scale")
    parser.add_argument("--scr", default="", help="path to score file")

    parser.add_argument("--pop", default=30, type=int, help="population")
    parser.add_argument("--n-gen", type=int, help="current generation index")
    args = parser.parse_args()
    return args

def evolution(args):
    logging.info("======================================================")
    logging.info("(Generation %d) Start generating genes..." % args.n_gen)
    logging.info(args)
    logging.info("======================================================")
    
    # make sure the order of the parameters is consistent with 
    # generated genes
    param_dict = OrderedDict(eval(args.params))
    map_func_dict = eval(args.map_func)

    if args.n_gen == 0:
        init_vec = []
        # convert values of parameters to be tuned into a vector    
        for param in param_dict.keys():
            assert param in map_func_dict
            init_vec.append(funcs[map_func_dict[param]](param_dict[param]))

        es = cma.CMAEvolutionStrategy(init_vec, 0.1, {
            'seed': 1,
            'popsize': args.pop,
        })
        X = es.ask()

    else:
        # load previous checkpoint
        with open(args.checkpoint % str(args.n_gen - 1).zfill(2), "rb") as es_file:
            es = pickle.load(es_file)

        X = es.ask()
        # open score file
        with open(args.scr) as f:
            # read score file
            scores = f.readlines()
        # get perplexity-val for each model as metric
        Y = list(map(lambda x: -float(x.strip().split("\t")[-1]), scores))
        es.tell(X, Y)

    # save current checkpoint
    with open(args.checkpoint % str(args.n_gen).zfill(2), "wb") as es_file:
        pickle.dump(es, es_file)

    # save generated genes for current generation
    for gene_idx in range(args.pop):
        with open(args.gene % str(gene_idx).zfill(2), "w") as gene:
            # map scaled genes to original values
            genes = [inverse_funcs[param](g) for param, g in zip(map_func_dict.values(), X[gene_idx])]
            gene.write("\n".join([p+"="+str(g) for p,g in zip(param_dict.keys(), genes)]))


if __name__ == "__main__":
    args = get_arguments()
    evolution(args)
    logging.info("Generated %s genes." % args.pop)
