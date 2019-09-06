## Multitarget TED Talks 

This recipe builds systems from the IARPA MATERIAL datasets:
Hyperparameter (hpm) files with well-tuned values are provided here. 
The resulting trained models are available at http://www.cs.jhu.edu/~kevinduh/a/sockeye-recipes-models/egs/iarpa-material 
You will need to point to your own copy of the data if you want to train from scratch. 


TODO (revise below)

### 1. Setup

First, download the data. It's about 575MB compressed and 1.8GB uncompressed.
```bash
sh ./0_download_data.sh
```

Then, setup the task for the language you are interested.
The following command creates a new working directory (zh-en) 
and populates it with several hyperparameter files 

```bash
sh ./1_setup_task.sh zh
cd zh-en
ls
```

You should see files like `rs1.hpm` which is one of the hyperparameter files we will run with. This file specifies a BPE symbol size of 30k for source and target, 512-dim word embeddings, 512-dim LSTM hiddent units in a 1-layer seq2seq network. Further, the checkpoint frequency is 4000 updates and all model information will be saved in ./rs1.

### 2. Preprocessing and Training

First, make sure we are in the correct working directory.

```bash
pwd
```

All hyperparameter files and instructions below assume we are in `$rootdir/egs/ted/zh-en`, where `$rootdir` is the location of the sockeye-recipes installation. 


Now, we can preprocess the tokenized training and dev data using BPE.
```bash
../../../scripts/preprocess-bpe.sh rs1.hpm
```

The resulting BPE vocabulary file (for English) is: `data-bpe/train.bpe-30000.en.bpe_vocab` and the segmented training file is: `data-bpe/train.bpe-30000.en`. For Chinese, replace `en` by `zh`. These are the files we train on. 

To train, we will use qsub and gpu (On a GeForce GTX 1080 Ti, this should take about 4 hours):

```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=12:00:00,num_proc=2 -j y ../../../scripts/train.sh -p rs1.hpm -e sockeye_gpu
```

Alternatively, if using local cpu:
```bash
../../../scripts/train.sh -p rs1.hpm -e sockeye_cpu
```

For the transformer models (e.g. tm1.hpm), there is a different training script, train-alloptions.sh instead of train.sh:

```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=12:00:00,num_proc=2 -j y ../../../scripts/train-alloptions.sh -p tm1.hpm -e sockeye_gpu
```


### 3. Evaluation

Again, make sure we are in the correct working directory (`$rootdir/egs/ted/zh-en`).

```bash
pwd
```

The test set we want to translate is `../multitarget-ted/en-zh/tok/ted_test1_en-zh.tok.zh`. We translate it using rs1 via qsub on gpu (this should take 10 minutes or less):

```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=00:30:00 -j y ../../../scripts/translate.sh -p rs1.hpm -i ../multitarget-ted/en-zh/tok/ted_test1_en-zh.tok.zh -o rs1/ted_test1_en-zh.tok.en.1best -e sockeye_gpu
```

Alternatively, to translate using local cpu:

```bash
../../../scripts/translate.sh -p rs1.hpm -i ../multitarget-ted/en-zh/tok/ted_test1_en-zh.tok.zh -o rs1/ted_test1_en-zh.tok.en.1best -e sockeye_cpu
```

When this is finished, we have the translations in `rs1/ted_test1_en-zh.tok.en.1best`. We can now compute the BLEU score by:

```bash
../../../tools/multi-bleu.perl ../multitarget-ted/en-zh/tok/ted_test1_en-zh.tok.en < rs1/ted_test1_en-zh.tok.en.1best
```

This should give a BLEU score of around 10.58.


### 4. Train systems for different language pairs

Let's repeat the above steps (1-3) on Arabic (ar).
First return to the parent directory (`$rootdir/egs/ted/`) and setup the task.

```bash
cd ../ 
sh ./1_setup_task.sh ar
cd ar-en
```

Now, we run preprocess and train: 

```bash
../../../scripts/preprocess-bpe.sh rs1.hpm
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=12:00:00,num_proc=2 -j y ../../../scripts/train.sh -p rs1.hpm -e sockeye_gpu
```

Finally, we translated and measure BLEU:

```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=00:30:00 -j y ../../../scripts/translate.sh -p rs1.hpm -i ../multitarget-ted/en-ar/tok/ted_test1_en-ar.tok.ar -o rs1/ted_test1_en-ar.tok.en.1best -e sockeye_gpu
../../../tools/multi-bleu.perl ../multitarget-ted/en-ar/tok/ted_test1_en-ar.tok.en < rs1/ted_test1_en-ar.tok.en.1best
```





