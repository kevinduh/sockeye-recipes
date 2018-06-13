## Quickstart: sockeye-recipes


We will train a model on a very small sample German-English data, just to confirm our installation works. The whole process should take less than 30 minutes. Since the data is so small, you should not expect the model to learn anything.

(1) Download and unpack the data in any directory. Let's use the current directory `$rootdir/egs/quickstart` as working directory. Note that `$rootdir` is defined as the base directory of the sockeye-recipes repository. 

```bash
cd $rootdir/egs/quickstart/
./0_download_data.sh
```

(2) Peruse the hyperparameters file. This is the file that specifies everything about an NMT run. The example hyperparameters file `tiny_rnn.hpm` has been prepared for you

```bash
cat tiny_rnn.hpm
```

Note that `workdir=./` so everything will be placed in the current directory. The relative path `$rootdir=../../` should point to your base directory in the sockeye-recipes repo; this can also be edited to a hardcoded path. 

(3) Preprocess data with BPE segmentation.

Run the `preprocess-bpe.sh` script, which will read the training/validation tokenized bitext (via `train_tok` and `valid_tok`), learn the BPE subword units (the number of which is specified by `bpe_symbols_src` and `bpe_symbols_trg`) and save the BPE'd text in `datadir=$workdir/data-bpe`

```bash
bash path/to/sockeye-recipes/scripts/preprocess-bpe.sh tiny_rnn.hpm
```

In practice, you can just run in the current directory:
```bash
bash ../../scripts/preprocess-bpe.sh tiny_rnn.hpm
  2018-06-01 15:37:41 - Learning BPE on source and creating vocabulary: .//data-bpe//train.bpe-4000.de.bpe_vocab
  2018-06-01 15:37:54 - Applying BPE, creating: .//data-bpe//train.bpe-4000.de, .//data-bpe//valid.bpe-4000.de
  2018-06-01 15:37:58 - Learning BPE on target and creating vocabulary: .//data-bpe//train.bpe-4000.en.bpe_vocab
  2018-06-01 15:38:07 - Applying BPE, creating: .//data-bpe//train.bpe-4000.en, .//data-bpe//valid.bpe-4000.en
  2018-06-01 15:38:10 - Done with preprocess-bpe.sh
```

This is a standard way (though not the only way) to handle large vocabulary in NMT. Currently sockeye-recipes assumes BPE segmentation before training. The preprocess-bpe.sh script takes a hyperparams file as input and preprocesses accordingly. To get a flavor of BPE segmentation results (train.en is original, train.bpe-4000.en is BPE'ed, and the string '@@' indicates BPE boundary):

```bash
head -3 sample-de-en/train.en data/train.bpe-4000.en
```

(4a) Now, we can train the NMT model. Generally, the invocation is:

```bash
bash path/to/sockeye-recipes/scripts/train -p HYPERPARAMETER_FILE -e SOCKEYE_ENVIRONMENT
```
The hyperparameter file specifies the model architecture and training data, while the Sockeye Conda Environment specifies the actual code and whether to run on CPU or GPU.

First, let's try the CPU version:

```bash
bash ../../scripts/train.sh -p tiny_rnn.hpm -e sockeye_cpu
```

The `train.sh` script starts of the training process. The `-p` flag indicates the hyperparameter file, and the `-e` flag indicates the sockeye environment you use to use. Whether we should run in CPU or GPU mode is inferred by what is installed in the environment. We now give it the `sockeye_cpu` environment, which was installed as a CPU environment. The model and all training info are saved in `modeldir=tiny_rnn/`. 

(4b) Second, let's try the GPU version. This assumes your machine has NVIDIA GPUs. First, we modify the `$modeldir` in the hyperparameter, to keep the training information separate. Next we run the same train.sh script but telling it to use the GPU environment (`sockeye_gpu`):

```bash
sed "s|tiny_rnn|tiny_rnn_gpu|" tiny_rnn.hpm > tiny_rnn_gpu.hpm
bash ../../scripts/train.sh -p tiny_rnn_gpu.hpm -e sockeye_gpu
```

Various sockeye-recipe scripts call scripts/get-device.sh to determine which device to use. If you are having trouble with the GPU run, check this script to see if it matches your computer system. 

Alternatively, all these commands can also be used in conjunction with the Sun/Univa Grid Engine, e.g.:

```
sed "s|tiny_rnn|tiny_rnn_gpu|" tiny_rnn.hpm > tiny_rnn_gpu.hpm
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=24:00:00,num_proc=2 -j y path/to/sockeye-recipes/scripts/train.sh -p tiny_rnn_gpu.hpm -e sockeye_gpu
(or)
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=24:00:00,num_proc=2 -j y ../../scripts/train.sh -p tiny_rnn_gpu.hpm -e sockeye_gpu
```


For Multi-GPU training, you can set `-l gpu=1` to a larger number `-l gpu=2`; the script relies on CUDA_VISIBLE_DEVICES to pick the free GPU cards. It is strongly recommended that CUDA_VISIBLE_DEVICES is set in your system; if not, the script will pick a single free GPU based on nvidia-smi (but this is not guaranteed to be safe in a multi-user enivornment).

Note that we specify `num_proc=2` in the `qsub` command. `train.sh` is setup to run one process for training (which may be on both CPU and GPU), and another process for decoding the validation set during checkpoints. Validation decoding is spawned as a separate CPU process, thus `num_proc=2`. Sockeye itself is flexible in validation decoding running in either CPU or GPU mode, but sockeye-recipe forces it to run in CPU mode only. This is a design decision meant to maintain efficient resource utilization in a shared computing grid. 

(5) Finally, we can translate new test sets with `translate.sh`:

Generally, the invocation is:
```bash
bash path/to/sockeye-recipes/scripts/translate.sh -i INPUT_SOURCE_FILE -o OUTPUT_TRANSLATION_FILE -p HYPERPARAMETER_FILE -e ENV
```

CPU version: 
```bash
bash ../../scripts/translate.sh -i sample-de-en/valid.de -o tiny_rnn/valid.en.1best -p tiny_rnn.hpm -e sockeye_cpu
```

GPU version: 
```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=00:50:00 -j y ../../scripts/translate.sh -i sample-de-en/valid.de -o tiny_rnn_gpu/valid.en.1best -p tiny_rnn_gpu.hpm -e sockeye_gpu
```

For GPU Sockeye environments, we can also force it to run on the CPU with the `-d cpu` flag: 
```bash
../../scripts/translate.sh -i sample-de-en/valid.de -o tiny_rnn/valid.en.1best -p tiny_rnn.hpm -e sockeye_gpu -d cpu
```

This `translate.sh` script will find the model from hyperparams file. Then it runs BPE on the input (which is assumed to be tokenized in the same way as train_tok and valid_tok), translates the result, runs de-BPE and saves in output.

(6) To visualize the learning curve, you can use tensorboard:

```bash
source activate sockeye_cpu
tensorboard --logdir ./

Then follow the instructions, e.g. pointing your browser to http://localhost:6006 . Note that not all features of Google's tensorboard is implemented in this DMLC MXNet port, but at least you can currently visualize perplexity curves and a few other useful things. 
```

All results are stored in the `$modeldir`. The ones of interest:

```bash
ls tiny_rnn/*
 ... 
 tiny_rnn/cmdline.log --> log of the sockeye-recipes script invocation     
 tiny_rnn/hyperparams.txt --> a backup copy of the hpm file (should be same as tiny_rnn.hpm)
 tiny_rnn/log --> log of sockeye training
 tiny_rnn/metrics --> records perplexity, time, etc. at each checkpoint
 tiny_rnn/params.00000 --> saved model at each checkpoint (e.g. 0)
 tiny_rnn/params.00002 --> saved model at each checkpoint (e.g. 2)
 tiny_rnn/params.best --> points to best model according to validation set
 tiny_rnn/tensorboard/ --> event logs for tensorboard visualization
 tiny_rnn/vocab.{src,trg}.0.json --> vocabulary file generated by sockeye
```