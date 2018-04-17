# sockeye-recipes
Training scripts and recipes for the Sockeye Neural Machine Translation (NMT) toolkit
- The original Sockeye codebase is at [AWS Labs](https://github.com/awslabs/sockeye)
- This version is based off [a stable fork](https://github.com/kevinduh/sockeye). The current sockeye version that sockeye-recipes is built on is: 1.18.1. 

This package contains scripts that makes it easy to run NMT experiments.
The way to use this package is to specify settings in a file like "hyperparams.txt", 
then run the following scripts:
- scripts/preprocess-bpe.sh: Preprocess bitext via BPE segmentation
- scripts/train.sh: Train the NMT model given bitext
- scripts/plot-validation-curve.sh: Compute BLEU curves by iteration on validation data
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
The third install_tools.sh script simply installs some helper tools, such as BPE preprocesser.

#### Re-Install

When the sockeye version is updated, it is recommended to re-run the installation scripts in a clean conda environment:

```bash
conda remove --name sockeye_gpu --all
conda remove --name sockeye_cpu --all
bash ./install/install_sockeye_cpu.sh
bash ./install/install_sockeye_gpu.sh
```

#### Environment Setup
To set up the running environment, we add the following configurations in the ~/.bashrc file.

Configure CUDA and CuDNN for the GPU version of Sockeye:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

Set up a clean UTF-8 environment to avoid encoding errors:

```bash
export LANG=en_US.UTF-8
```


## Quick Example Run
We will train a model on a very small sample German-English data, just to confirm our installation works. The whole process should take less than 30 minutes. Since the data is so small, you should not expect the model to learn anything. 

(1) Download and unpack the data in any directory. Let's make our working directory "sockeye_trial" in this example:
```bash
mkdir ~/sockeye_trial
cd ~/sockeye_trial
wget https://cs.jhu.edu/~kevinduh/j/sample-de-en.tgz
tar -xzvf sample-de-en.tgz
```

(2) Edit the hyperparams.txt file. We can use the example in examples/hyperparams.sample-de-en.txt. First, copy it to your current working directory:

```bash
cd ~/sockeye_trial
cp path/to/sockeye-recipes/examples/hyperparams.sample-de-en.txt .
```

Then, please open up an editor and edit the "rootdir" setting in hyperparams.sample-de-en.txt
to point to your sockeye-recipes installation path, e.g. ~/src/sockeye-recipes
Note that this hyperparms file specifies all of your file/script locations and model training configurations, and is the recipe for every experiment. 
The other settings in the example can be used as is, but if your paths have changed, make sure to modify workdir, datadir, modeldir accordingly. See the file for detailed explanation.

(3) Preprocess data with BPE segmentation. 

```bash
bash path/to/sockeye-recipes/scripts/preprocess-bpe.sh hyperparams.sample-de-en.txt
```

This is a standard way (though not the only way) to handle large vocabulary in NMT. Currently sockeye-recipes assumes BPE segmentation before training. The preprocess-bpe.sh script takes a hyperparams file as input and preprocesses accordingly. To get a flavor of BPE segmentation results (train.en is original, train.bpe-4000.en is BPE'ed, and the string '@@' indicates BPE boundary): 

```bash
head -3 sample-de-en/train.en data/train.bpe-4000.en
```

(4) Now, we can train the NMT model. We give the train.sh script the hyperparameters and tell it whether to train on CPU or GPU.

First, let's try the CPU version:
```bash
bash path/to/sockeye-recipes/scripts/train.sh hyperparams.sample-de-en.txt cpu
```

The model and all training info are saved in modeldir (~/sockeye_trial/model1).

Optionally, let's try GPU version. This assumes your machine has NVIDIA GPUs. First, we modify the modeldir hyper-parameter to model2, to keep the training information separate. Next we run the same train.sh script but telling it to use the gpu:
```bash
sed 's/model1/model2/' hyperparams.sample-de-en.txt > hyperparams.sample-de-en.2.txt
bash path/to/sockeye-recipes/scripts/train.sh hyperparams.sample-de-en.2.txt gpu
```

The GPU version calls scripts/get-gpu.sh to find a free GPU card on the current machine. Sockeye allows multi-GPU training but in these recipes we only use one GPU per training process. 

Alternatively, all these commands can also be used in conjunction with Univa Grid Engine, e.g.:
```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=24:00:00 -j y -o train.log path/to/sockeye-recipes/scripts/train.sh hyperparams.sample-de-en.2.txt gpu
```

(5) We can measure how BLEU changes per iteration on the validation data with one of the following:


```bash
# CPU version on local machine:
bash path/to/sockeye-recipes/scripts/plot-validation-curve.sh hyperparams.sample-de-en.txt cpu
# GPU version on local machine:
bash path/to/sockeye-recipes/scripts/plot-validation-curve.sh hyperparams.sample-de-en.txt gpu
# GPU version with Univa Grid Engine:
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=24:00:00 -j y -o valid.log path/to/sockeye-recipes/plot-validation-curve.sh hyperparams.sample-de-en.2.txt gpu
```

After this finishes running, you can see the translation outputs in out.valid_bpe.* and corresponding BLEU scores in multibleu.valid_bpe.result, in the modeldir. 
Note that for quick demonstration, this example uses very small data and very short training time. Mostly likely the translation will be junk and BLEU will be close to zero. 

(6) Finally, we can translate new test sets with:

```bash
bash path/to/sockeye-recipes/scripts/translate.sh hyperparams.sample-de-en.2.txt input output device(cpu/gpu)
```

This script will find the model from hyperparams file. Then it runs BPE on the input (which is assumed to be tokenized in the same way as train_tok and valid_tok), translates the result, runs de-BPE and saves in output. 


(7) To visualize the learning curve, you can use tensorboard:

```bash
source activate sockeye_cpu
tensorboard --logdir ~/sockeye_trial/model1
```

Then follow the instructions, e.g. pointing your browser to http://localhost:6006 . Note that not all features of Google's tensorboard is implemented in this DMLC MXNet port, but at least you can currently visualize perplexity curves and a few other things.  


## Full Example Run (WMT14 English-German)

This example trains on a full English-German dataset of 4.5 million sentence pairs, drawn from WMT14 and packaged by <a href="https://nlp.stanford.edu/projects/nmt/">Stanford</a>. The results should be comparable to the <a href="https://nlp.stanford.edu/pubs/emnlp15_attn.pdf">Luong EMNLP2015 paper</a>.

We will copy over the script "wmt14-en-de.sh" and hyperparameter file "hyperparams.wmf14-en-de.txt" to your working directory "sockeye_trial2". The script downloads the data, runs BPE preprocessing and starts off a training process via qsub:

```bash
mkdir sockeye_trial2
cp examples/wmt14-en-de.sh sockeye_trial2/
cp examples/hyperparams.wmt14-en-de.txt sockeye_trial2/
cd sockeye_trial2
```

Before running wmt14-en-de.sh, make sure to modify rootdir in the hyperparameters file to point to your sockeye-recipes directory. If you saved the data in a different directory, make sure to modify train_tok and valid_tok. If you are using a different workdir to keep results, modify that in the hyperparameters file too. According to the hyperparams.wmt14-en-de.txt file, note the model will be trained on train.{en,de} (train_tok) and validated on newstest2013 (valid_tok). 

```bash
bash wmt14-en-de.sh
```

The downloading/preprocessing part may take up to 2 hours, and training may take up to 20 hours (on a GTX 1080 Ti GPU) for this particular hyperparameter configuration. The BLEU score on newstest2014.de after decoding should be around 19.33.

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
