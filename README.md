# sockeye-recipes

Training scripts and recipes for the Sockeye Neural Machine Translation (NMT) toolkit
- The original Sockeye codebase is at [AWS Labs](https://github.com/awslabs/sockeye)
- This version is based off [a stable fork](https://github.com/kevinduh/sockeye). The current sockeye version that sockeye-recipes is built on is: 1.18.57. 

This package contains scripts that makes it easy to run NMT experiments.
The way to use this package is to specify settings in a file like "hyperparams.txt", 
then run the following scripts:
- scripts/preprocess-bpe.sh: Preprocess bitext via BPE segmentation
- scripts/train.sh: Train the NMT model given bitext
- scripts/translate.sh: Translates a tokenized input file using an existing model


## Installation
First, clone this package: 
```bash
git clone https://github.com/kevinduh/sockeye-recipes.git sockeye-recipes
```

We assume that Anaconda for Python virtual environments is available on the system.
Run the following to install Sockeye in two Anaconda environments, sockeye_cpu and sockeye_gpu: 

```bash
cd path/to/sockeye-recipes
bash ./install/install_sockeye_cpu.sh
bash ./install/install_sockeye_gpu.sh
bash ./install/install_tools.sh
```

The training scripts and recipes will activate either the sockeye_cpu or sockeye_gpu environment depending on whether CPU or GPU is specified. 
Currently we assume CUDA 9.0 is available for GPU mode; this can be changed if needed. 
The third install_tools.sh script simply installs some helper tools, such as BPE preprocesser.

#### Re-Install

When the sockeye version is updated, it is recommended to re-run the installation scripts in a clean conda environment:

```bash
conda remove --name sockeye_cpu --all
conda remove --name sockeye_gpu --all
bash ./install/install_sockeye_cpu.sh
bash ./install/install_sockeye_gpu.sh
```

If you want to back-up an existing version of the conda environment, run this before re-installing:

```bash
conda create --name sockeye_gpu_bkup --clone sockeye_gpu
conda create --name sockeye_cpu_bkup --clone sockeye_cpu
```

#### Environment Setup
Depending on your computer setup, you may want add the following configurations in the ~/.bashrc file.

Configure CUDA and CuDNN for the GPU version of Sockeye:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/NVIDIA/cuda-9.0/lib64
```

Set up a clean UTF-8 environment to avoid encoding errors:

```bash
export LANG=en_US.UTF-8
```

## Recipes 

The `egs` subdirectory contains recipes for various datasets. 

* [egs/quickstart](egs/quickstart): For first time users, this recipe explains how sockeye-recipe works. 

* [egs/ted](egs/ted): Recipes for training various NMT models, using a TED Talks dataset consisting of 20 different languages. 

* [egs/wmt14-en-de](egs/wmt14-en-de): Recipe for training a baseline that compares with the <a href="https://nlp.stanford.edu/pubs/emnlp15_attn.pdf">Luong EMNLP2015 paper</a>.

* [egs/curriculum](egs/curriculum): Recipe for curriculum learning. Also explains how to use sockeye-recipes in conjunction with a custom sockeye installation.

* [egs/optimizers](egs/optimizers): Example of training with different optimizers (e.g. ADAM, EVE, Nesterov ADAM, SGD, ...)

The [hpm](hpm) subdirectory contains hyperparameter (hpm) file templates. Besides NMT hyerparameters, the most important variables in this file to set are below: 

* rootdir: location of your sockeye-recipes installation, used for finding relevant scripts (i.e. this is current directory, where this README file is located.)

* modeldir: directory for storing a single Sockeye model training process

* workdir: directory for placing various modeldirs (i.e. a suite of experiments with different hyperparameters) corresponding to the same dataset

* train_tok and valid_tok: prefix of tokenized training and validation bitext file path

* train_bpe_{src,trg} and valid_bpe_{src,trg}: alternatively, prefix of the above training and validation files already processed by BPE


## Auto-Tuning ##

This package also provides tools for auto-tuning, where one can specify the hyperparameters to search over and a meta-optimizer automatically attempts to try different configurations that it believes will be promising. For more information, see the auto-tuning folder. 


## Design Principles and Suggested Usage

Building NMT systems can be a tedious process involving lenghty experimentation with hyperparameters. The goal of sockeye-recipes is to make it easy to try many different configurations and to record best practices as example recipes. The suggested usage is as follows:
- Prepare your training and validation bitext beforehand with the necessary preprocessing (e.g. data consolidation, tokenization, lower/true-casing). Sockeye-recipes simply assumes pairs of train_tok and valid_tok files. 
- Set the working directory to correspond to a single suite of experiments on the same dataset (e.g. WMT17-German-English)
- The only preprocessing handled here is BPE. Run preprocess-bpe.sh with different BPE vocabulary sizes (bpe_symbols_src, bpe_symbols_trg). These can be saved all to the same datadir.
- train.sh is the main training script. Specify a new modeldir for each train.sh run. The hyperparms.txt file used in training will be saved in modeldir for future reference. 
- At the end, your workingdir will have a single datadir containing multiple BPE'ed versions of the bitext, and multiple modeldir's. You can run tensorboard on all these modeldir's concurrently to compare learning curves.

There are many options in Sockeye. Currently not all of them are used in sockeye-recipes; more will be added. See [sockeye/arguments.py](https://github.com/kevinduh/sockeye/blob/master/sockeye/arguments.py) for detailed explanations. 

Alternatively, directly call sockeye with the help option as below. Note that sockeye-recipe hyperameters have the same name as sockeye hyperparameters, except that sockeye-recipe hyperparameters replace the hyphen with underscore (e.g. --num-embed in sockeye becomes $num_embed in sockeye-recipes):
 
```bash
source activate sockeye_cpu
python -m sockeye.train --help
```
