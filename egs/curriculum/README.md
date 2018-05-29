# Curriculum Learning example for Sockeye
Training script and recipes for Sockeye.
This implementation makes the following assumptions:
- The curriculum is determined by discrete complexity classes (0=easy, 1=a bit harder, 2=even harder, higher is harder)
- The curriculum will only expose the model to easy data when training starts and will gradually increase the hardness (complexity) of the data.
- Exactly when curriculum increases hardness can be controlled using the `--curriculum-update-freq` switch in training,
- Uses shards (originally meant for data parallelism) to split the data by hardness (complexity)


#### Install custom sockeye with curriculum learning enabled
Checkout the `curriculum` branch of a custom sockeye from here
```bash
cd path/to/sockeye-recipes
git clone https://github.com/noisychannel/sockeye.git sockeye-curriculum
cd sockeye-curriculum
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
cd path/to/sockeye-recipes
bash ./install/install_sockeye_custom -s ./sockeye-curriculum -e sockeye-curriculum
```

A list of your environments may be obtained by running `conda info --envs`

#### Re-Install

You may choose to make your own changes to the sockeye installation. To bring these changes into the conda environment, run the installation script again.
Make sure that the `ENV_NAME` is the same as before, e.g.:
```bash
cd path/to/sockeye-recipes
bash ./install/install_sockeye_custom -s ./sockeye-curriculum-with-changes -e sockeye-curriculum
```

## Quick Example Run

This example uses a small subsample of TED German-English dataset for demo purposes.
The whole process should take less than 30 minutes. Since the data is so small, you should not expect the model to learn anything. 

(1) Let's do everything in the egs/curriculum directory. First, let's get the data:
```bash
cd path/to/sockeye-recipes/egs/curriculum
wget https://cs.jhu.edu/~kevinduh/j/sample-de-en.tgz
tar -xzvf sample-de-en.tgz
```

The corresponding curriculum complexity scores are defined in `curriculum_sent.scores`.
The latter file is the discretized score (complexity) per sentence.

(2) Preprocess data with BPE segmentation. 
If you haven't done this before, BPE the sample data.
Note the hyperparametes file `hyperparams.curr-de-en.sample.txt` has set up relative paths for `workdir` and `rootdir`, which should work from this directory (`egs/curriculum`). 

```bash
bash ../../scripts/preprocess-bpe.sh hyperparams.curr-de-en.sample.txt
```

(3) Now, we can train the NMT model. We will use `train-curriculum.sh` in `sockeye-recipes/scripts`. This assumes that you have a machine with an available GPU.

```bash
bash path/to/sockeye-recipes/scripts/train-curriculum.sh hyperparams.curr-de-en.sample.txt ENV_NAME
```
Replace `ENV_NAME` with the name of the environment from the installation process. E.g.,
```bash
bash ../../scripts/train-curriculum.sh hyperparams.curr-de-en.sample.txt sockeye-curriculum
```

Alternatively, all these commands can also be used in conjunction with Univa Grid Engine, e.g.:
```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=24:00:00 -j y -o train.log path/to/sockeye-recipes/scripts/train-curriculum.sh hyperparams.curr-de-en.sample.txt sockeye-curriculum
```

(4) Validation and evaluation proceeds as usual.

(5) Examining the training logs
The logs for curriculum training will indicate when the curriculum is updated and which shards are visible for training based on this constraint.
```bash
[INFO:sockeye.data_io] ** Updating complexity constraint (increased by 1)
[INFO:sockeye.data_io] **** Old max complexity 1
[INFO:sockeye.data_io] **** New max complexity 2
[INFO:sockeye.data_io] Shards visible based on complexity constraint are: /exp/gkumar/exp/sockeye_trial//model_curr1/prepared_data/shard.00000,/exp/gkumar/exp/sockeye_trial//model_curr1/prepared_data/sh
ard.00001,/exp/gkumar/exp/sockeye_trial//model_curr1/prepared_data/shard.00002
```
