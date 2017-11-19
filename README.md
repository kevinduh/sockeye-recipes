# sockeye-recipes
Training scripts and recipes for the Sockeye Neural Machine Translation (NMT) toolkit
- The original Sockeye codebase is at [AWS Labs](https://github.com/awslabs/sockeye)
- This version is based off [a stable fork](https://github.com/kevinduh/sockeye)

This package contains scripts that makes it easy to run NMT experiments.
The way to use this package is to specify settings in a file like "hyperparams.txt", 
then run the following scripts:
- scripts/preprocess-bpe.sh: Preprocess bitext via BPE segmentation
- scripts/train.sh: Train the NMT model given bitext
- scripts/plot-validation-curve.sh: Compute BLEU curves by iteration on validation data

## Installation
First, clone this package: 
```bash
git clone https://github.com/kevinduh/sockeye-recipes.git sockeye-recipes
```

We assume that Anaconda for Python virtual environments is available on the system.
Run the following to install Sockeye in two Anaconda environments, sockeye_cpu and sockeye_gpu: 

```bash
cd path/to/sockeye-recipes
sh ./install/install_sockeye_cpu.sh
sh ./install/install_sockeye_gpu.sh
sh ./install/install_tools.sh
```

The training scripts and recipes will activate either the sockeye_cpu or sockeye_gpu environment depending on whether CPU or GPU is specified. 
The third install_tools.sh script simply installs some helper tools, such as BPE preprocesser.


## Example Run
We will train a model on some sample German-English data.

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
sh path/to/sockeye-recipes/scripts/preprocess-bpe.sh hyperparams.sample-de-en.txt
```

This is a standard way (though not the only way) to handle large vocabulary in NMT. Currently sockeye-recipes assumes BPE segmentation before training. The preprocess-bpe.sh script takes a hyperparams file as input and preprocesses accordingly. To get a flavor of BPE segmentation results (train.en is original, train.bpe-4000.en is BPE'ed, and the string '@@' indicates BPE boundary): 

```bash
head -3 sample-de-en/train.en sample-de-en/train.bpe-4000.en
```

(4) Now, we can train the NMT model. We give the train.sh script the hyperparameters and tell it whether to train on CPU or GPU.

First, let's try the CPU version:
```bash
sh path/to/sockeye-recipes/scripts/train.sh hyperparams.sample-de-en.txt cpu
```

The model and all training info are saved in modeldir (~/sockeye_trial/model1).

Optionally, let's try GPU version. This assumes your machine has NVIDIA GPUs. First, we modify the modeldir hyper-parameter to model2, to keep the training information separate. Next we run the same train.sh script but telling it to use the gpu:
```bash
sed 's/model1/model2/' hyperparams.sample-de-en.txt > hyperparams.sample-de-en.2.txt
sh path/to/sockeye-recipes/scripts/train.sh hyperparams.sample-de-en.2.txt gpu
```

The GPU version calls scripts/get-gpu.sh to find a free GPU card on the current machine. Sockeye allows multi-GPU training but in these recipes we only use one GPU per training process. 

Alternatively, all these commands can also be used in conjunction with Univa Grid Engine, e.g.:
```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=24:00:00 -j y -o train.log path/to/sockeye-recipes/scripts/train.sh hyperparams.sample-de-en.2.txt gpu
```

(5) We can measure how BLEU changes per iteration on the validation data with one of the following:


```bash
# CPU version on local machine:
sh path/to/sockeye-recipes/scripts/plot-validation-curve.sh hyperparams.sample-de-en.txt cpu
# GPU version on local machine:
sh path/to/sockeye-recipes/scripts/plot-validation-curve.sh hyperparams.sample-de-en.txt gpu
# GPU version with Univa Grid Engine:
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=24:00:00 -j y -o valid.log path/to/sockeye-recipes/plot-validation-curve.sh hyperparams.sample-de-en.2.txt gpu
```

After this finishes running, you can see the translation outputs in out.valid_bpe.* and corresponding BLEU scores in multibleu.valid_bpe.result, in the modeldir. 
Note that for quick demonstration, this example uses very small data and very short training time. Mostly likely the translation will be junk and BLEU will be close to zero. 

(6) Finally, we can translate new test sets with:

```bash
sh path/to/sockeye-recipes/scripts/translate.sh modeldir bpe_vocab_src input output device(cpu/gpu)
```

Note that this script does not require a hyperparams file. One simply needs to point to the modeldir and the BPE source vocabulary (usually in datadir) in order to start up a translation service.