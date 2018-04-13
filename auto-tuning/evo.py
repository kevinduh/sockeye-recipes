import cma
import numpy as np
import math
import pickle
import os
import argparse
from logger import *
from collections import OrderedDict
'''
range_dict = {'num_src_embed':(64,1024), 
             'num_trg_embed':(64,1024),
             'rnn_num_hidden':(64,2048),
             'num_layers':(1,2),
             'num_src_words':(2000,80000),
             'num_trg_words':(2000,80000),
             'word_src_count':(1,2),
             'word_trg_count':(1,2),
             'batch_size':(16,256),
             'embed_src_dropout':(0,0.8),
             'embed_trg_dropout':(0,0.8),
             'rnn_encoder_dropout_outputs':(0,0.8),
             'rnn_decoder_dropout_outputs':(0,0.8),
             'rnn_encoder_dropout_states':(0,0.8),
             'rnn_decoder_dropout_states':(0,0.8),
             'rnn_decoder_hidden_dropout':(0,0.8),
             'initial_learning_rate':(0.0001,0.1)   
             }
'''

trans_dict = {'num_src_embed': (int, lambda x : 2**(6+4*x)), 
             'num_trg_embed': (int, lambda x : 2**(6+4*x)),
             'rnn_num_hidden': (int, lambda x : int(2**(6+5*x)) if int((2**(6+5*x)))%2==0 else int((2**(6+5*x)))+1),
             'num_layers': (int, lambda x : 1 + x),
             'num_src_words': (int, lambda x : 2000+78000*x),
             'num_trg_words': (int, lambda x : 2000+78000*x),
             'word_src_count': (int, lambda x : 1 + x),
             'word_trg_count': (int, lambda x : 1 + x),
             'batch_size': (int, lambda x : 2**(4+4*x)),
             'embed_src_dropout': (float, lambda x : 0.8*x),
             'embed_trg_dropout': (float, lambda x : 0.8*x),
             'rnn_encoder_dropout_outputs': (float, lambda x : 0.8*x),
             'rnn_decoder_dropout_outputs': (float, lambda x : 0.8*x),
             'rnn_encoder_dropout_states': (float, lambda x : 0.8*x),
             'rnn_decoder_dropout_states': (float, lambda x : 0.8*x),
             'rnn_decoder_hidden_dropout': (float, lambda x : 0.8*x),
             'initial_learning_rate': (int, lambda x : 10**(-1-3*x))   
             }

reverse_trans_dict = {'num_src_embed': lambda x : (math.log(x,2)-6)/4., 
             'num_trg_embed': lambda x : (math.log(x,2)-6)/4.,
             'rnn_num_hidden': lambda x : (math.log(x,2)-6)/5.,
             'num_layers': lambda x : x - 1,
             'num_src_words': lambda x : (x-2000)/78000.,
             'num_trg_words': lambda x : (x-2000)/78000.,
             'word_src_count': lambda x : x - 1,
             'word_trg_count': lambda x : x - 1,
             'batch_size': lambda x : (math.log(x,2)-4)/4.,
             'embed_src_dropout': lambda x : x/0.8,
             'embed_trg_dropout': lambda x : x/0.8,
             'rnn_encoder_dropout_outputs': lambda x : x/0.8,
             'rnn_decoder_dropout_outputs': lambda x : x/0.8,
             'rnn_encoder_dropout_states': lambda x : x/0.8,
             'rnn_decoder_dropout_states': lambda x : x/0.8,
             'rnn_decoder_hidden_dropout': lambda x : x/0.8,
             'initial_learning_rate': lambda x : (math.log(x,10)+1)/(-3.)
             }

def get_arguments():
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument("--checkpoint", default="checkpoints/es_G%s.pkl",
                        help="path to checkpoint files")
    parser.add_argument("--gene", default="generation_%s/genes/%s.gene",
                        help="path to backup es files")
    parser.add_argument("--params", type=str,
                        help="hyper-parameters to be tuned with initial value")
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

    if args.n_gen == 0:
        init_vec = []

        # convert values of parameters to be tuned into a vector    
        for param in param_dict.keys():
            assert param in reverse_trans_dict
            init_vec.append(float(reverse_trans_dict[param](param_dict[param])))

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
            # get bleu-val or pareto level for each model as metric
            Y = [float(l.split("\t")[-1]) for l in scores]
        es.tell(X, Y)

    # save current checkpoint
    with open(args.checkpoint % str(args.n_gen).zfill(2), "wb") as es_file:
        pickle.dump(es, es_file)

    # save generated genes for current generation
    for gene_idx in range(args.pop):
        with open(args.gene % str(gene_idx).zfill(2), "w") as gene:
            # map scaled genes to original values
            chop = lambda x : 1 if x>1 else (0 if x<0 else x)
            genes = [trans_dict[param][1](chop(g)) if trans_dict[param]==float else int(round(trans_dict[param][1](chop(g))))
                                                   for param, g in zip(param_dict.keys(), X[gene_idx])]
            gene.write("\n".join([p+"="+str(g) for p,g in zip(param_dict.keys(), genes)]))


if __name__ == "__main__":
    args = get_arguments()
    evolution(args)
    logging.info("(Generation %d) Generated %s genes." % (args.n_gen,args.pop))
