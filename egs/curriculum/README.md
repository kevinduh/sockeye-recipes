# Curriculum training for sockeye
Training script and recipes for Sockeye.
This implementation makes the following assumptions:
- The curriculum is determined by discrete complexity classes (0=easy, higher is harder)
- The curriculum will only expose the model to easy data when training starts and will gradually increase the hardness (complexity) of the data.
- The curriculum update can be controlled using the `--curriculum-update-freq` switch in training
- Uses shards (originally meant for data parallelism) to split the data by hardness (complexity)

This example uses the TED de-en dataset for experiments.

## Installation
First, clone this package: 
```bash
git clone https://github.com/kevinduh/sockeye-recipes.git sockeye-recipes
```
Then, switch to the `mtma` branch
```bash
git checkout mtma
```

#### Install custom sockeye
Checkout the `curriculum` branch of sockeye from here
```bash
git clone https://github.com/noisychannel/sockeye.git
git checkout curriculum
```

Install this custom version of sockeye (will install the GPU version by default, CUDA 8.0)
```bash
cd path/to/sockeye-recipes
bash ./install/install_sockeye_custom -s [SOCKEYE_CURRICULUM_LOCATION] -e [ENV_NAME]
```

You may choose any `ENV_NAME`, but you will need to remember this for future experiments
E.g.,
```bash
cd /exp/gkumar/code/sockeye-recipe-mtma
bash ./install/install_sockeye_custom -s /exp/gkumar/exp/sockeye_curr -e curriculum
```

A list of your environments may be obtained by running `conda info --envs`

#### Re-Install

You may choose to make your own changes to the sockeye installation. To bring these changes into the conda environment, run the installation script again.
Make sure that the `ENV_NAME` is the same as before
```bash
cd /exp/gkumar/code/sockeye-recipe-mtma
bash ./install/install_sockeye_custom -s /exp/gkumar/exp/sockeye_curr -e curriculum
```

## Quick Example Run
This is adapated from the example in the root directory of `sockeye-recipes`. Will run curriculum training.

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
cp path/to/sockeye-recipes/egs/curriculum/hyperparams.curr-de-en.sample.txt .
cp path/to/sockeye-recipes/egs/curriculum/curriculum_sent.scores .
```

The latter file is the discretized score (complexity) per sentence.

Then, please open up an editor and edit the "rootdir" setting in hyperparams.sample-de-en.txt
to point to your sockeye-recipes installation path, e.g. ~/src/sockeye-recipes
Note that this hyperparms file specifies all of your file/script locations and model training configurations, and is the recipe for every experiment. 
The other settings in the example can be used as is, but if your paths have changed, make sure to modify workdir, datadir, modeldir accordingly. See the file for detailed explanation.

(3) Preprocess data with BPE segmentation. 
If you haven't done this before, BPE the sample data

```bash
bash path/to/sockeye-recipes/scripts/preprocess-bpe.sh hyperparams.curr-de-en.sample.txt
```

(4) Now, we can train the NMT model. We will use `train-curr.sh` in `sockeye-recipes/scripts`. This assumes that you have a machine with an available GPU.

```bash
bash path/to/sockeye-recipes/scripts/train-curr.sh hyperparams.curr-de-en.sample.txt ENV_NAME
```
Replace `ENV_NAME` with the name of the environment from the installation process. E.g.,
```bash
bash /exp/gkumar/code/sockeye-recipes/scripts/train-curr.sh hyperparams.curr-de-en.sample.txt curriculum
```

Alternatively, all these commands can also be used in conjunction with Univa Grid Engine, e.g.:
```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=24:00:00 -j y -o train.log path/to/sockeye-recipes/scripts/train-curr.sh hyperparams.curr-de-en.sample.txt curriculum
```

(5) Validation and evaluation proceeds as usual.

(6) Examining the training logs
The logs for curriculum training will indicate when the curriculum is updated and which shards are visible for training based on this constraint.
```bash
[INFO:sockeye.data_io] ** Updating complexity constraint (increased by 1)
[INFO:sockeye.data_io] **** Old max complexity 1
[INFO:sockeye.data_io] **** New max complexity 2
[INFO:sockeye.data_io] Shards visible based on complexity constraint are: /exp/gkumar/exp/sockeye_trial//model_curr1/prepared_data/shard.00000,/exp/gkumar/exp/sockeye_trial//model_curr1/prepared_data/sh
ard.00001,/exp/gkumar/exp/sockeye_trial//model_curr1/prepared_data/shard.00002
```
