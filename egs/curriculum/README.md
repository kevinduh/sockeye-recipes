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
Note that you can install the custom sockeye anywhere. It does not need to be under `$rootdir`.

#### Re-Install

You may choose to make your own changes to the sockeye installation. To bring these changes into the conda environment, run the installation script again.
Make sure that the `ENV_NAME` is the same as before, e.g.:
```bash
cd path/to/sockeye-recipes
bash ./install/install_sockeye_custom -s path/to/sockeye-curriculum-with-changes -e sockeye-curriculum
```


## Example run on TED de-en data

(1) First, we assume that the TED de-en data in the `$rootdir/egs/ted` recipes is already prepared. If not, please run the following:

```bash
[egs/curriculum] / cd ../ted/
[egs/ted] ./0_download_data.sh
[egs/ted] ./1_setup_task.sh de
[egs/ted] cd de-en
[egs/ted/de-en] ../../../scripts/preprocess-bpe.sh rs1.hpm
```

(2) Next, we can setup the curriculum learning task, starting in the `egs/curriculum` subdirectory. The following downloads an example curriculum score file that corresponds to the bitext in `egs/ted/de-en/data-bpe`, and setups the the hyperparameter file.

``bash
cd $rootdir/egs/curriculum
[egs/curriculum/] ./0_download_data.sh
[egs/curriculum/] ./1_setup_task.sh
[egs/curriculum/] cd de-en
[egs/curriculum/de-en] ls
 curriculum_sent.scores
 data-bpe
 rs1.hpm
```

The curriculum complexity scores are defined in `curriculum_sent.scores`.
This file is the discretized score (complexity) per sentence.
This example sets up the de-en model using hyperparameter file template `rs1` (rnn smll model 1). `0_download_data.sh` and `1_setup_task.sh` can be modified for other data or hyperparameter file.

Note the main change to the hyperparameter file is the addition of `score_file` and `curriculum_update_freq`. Curriculum learning results will depend very much on these two. Currently we successively increase the sampling of harder examples after every 1000 updates. 

```bash
[egs/curriculum/de-en] tail rs1.hpm
...
 # For curriculum learning
 score_file=${workdir}/curriculum_sent.scores
 curriculum_update_freq=1000
```

(3) Now, we can train the NMT model. We will use `train-curriculum.sh` in `sockeye-recipes/scripts`. This assumes that you have a machine with an available GPU.

```bash
../../../scripts/train-curriculum.sh -p rs1.hpm -e sockeye-curriculum
```

Alternatively, all these commands can also be used in conjunction with Univa Grid Engine, e.g.:
```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=24:00:00 -j y -o train.log ../../../scripts/train-curriculum.sh -p rs1.hpm -e sockeye-curriculum
```

(4) Validation and evaluation proceeds as usual. Hopefully the curriculum results in faster training or better translation compared to random minibatches. The result in `egs/curriculum/de-en/rs1` can be directly compared to `egs/ted/de-en/rs1`

(5) Examining the training logs
The logs for curriculum training will indicate when the curriculum is updated and which shards are visible for training based on this constraint.
```bash
[INFO:sockeye.data_io] ** Updating complexity constraint (increased by 1)
[INFO:sockeye.data_io] **** Old max complexity 1
[INFO:sockeye.data_io] **** New max complexity 2
[INFO:sockeye.data_io] Shards visible based on complexity constraint are: rs1/prepared_data/shard.00000,rs1/prepared_data/shard.00001,rs1/prepared_data/shard.00002
```
