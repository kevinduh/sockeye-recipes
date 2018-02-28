import os
import multiprocessing
from logger import *
import argparse

# command for submitting gpu task
qsub_sh = "qsub -sync y -l 'gpu=1,mem_free=12g,ram_free=12g' -q g.q " 

def get_arguments():
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument("--hyperparams", type=str, help="the hyperparams file")
    parser.add_argument("--pop", default=30, type=int, help="population")
    parser.add_argument("--num-devices", default=1, type=int, 
                        help="the number of computation resources allocated")
    parser.add_argument("--autotunedir", type=str, help="root directory of the auto-tuning scripts")
    parser.add_argument("--device", type=str, help="options for cpu vs gpu training")
    parser.add_argument("--generation-path", type=str, help="path to current generation folder")
    parser.add_argument("--gene",type=str,help="path to gene files")
    parser.add_argument("--n-generation", type=int, help="current generation")
    
    args = parser.parse_args()
    return args

def form_qsub(n_pop):
    gene = args.gene%(str(n_pop).zfill(2))
    params = [args.hyperparams, args.autotunedir, args.device, args.generation_path, str(n_pop), gene, str(args.n_generation)]
    train_sh = args.autotunedir + "/train.sh "
    return train_sh + " ".join(params)

def run_train_gpu(n_pop):
    logging.info("(Generation %d) Start training model %s ......"%(args.n_generation, str(n_pop)))
    metrics_file = os.path.join(args.generation_path, "model_%s/metrics"%(str(n_pop).zfill(2)))
    # if the "training_state" file exists in the model directory
    # then it implies the model has not finished training
    state_file = os.path.join(args.generation_path, "model_%s/training_state"%(str(n_pop).zfill(2)))
    #log_file = os.path.join(args.generation_path, "model_%s/log"%(str(n_pop).zfill(2)))
    while((not os.path.exists(metrics_file)) or (os.path.exists(state_file))): #os.popen("tail -1 %s"%log_file).read().startwith("OSError"))
        os.system(qsub_sh + form_qsub(n_pop))
    logging.info("(Generation %d) Finish training model %s ......"%(args.n_generation, str(n_pop)))

def run_train_cpu(n_pop):
    logging.info("(Generation %d) Start training model %s ......"%(args.n_generation, (str(n_pop))))
    metrics_file = os.path.join(args.generation_path, "model_%s/metrics"%(str(n_pop).zfill(2)))
    state_file = os.path.join(args.generation_path, "model_%s/training_state"%(str(n_pop).zfill(2)))
    #log_file = os.path.join(args.generation_path, "model_%s/log"%(str(n_pop).zfill(2)))
    while((not os.path.exists(metrics_file)) or (os.path.exists(state_file))):
        os.system("sh " + form_qsub(n_pop))
    logging.info("(Generation %d) Finish training model %s ......"%(args.n_generation, (str(n_pop))))

def train_parallel(num_devices, pop, device):
    pool = multiprocessing.Pool(num_devices)
    if device=="cpu":
        pool.map(run_train_cpu, [n_pop for n_pop in range(pop)])
    else:
        pool.map(run_train_gpu, [n_pop for n_pop in range(pop)])


if __name__ == "__main__":
    args = get_arguments()
    train_parallel(args.num_devices, args.pop, args.device)
