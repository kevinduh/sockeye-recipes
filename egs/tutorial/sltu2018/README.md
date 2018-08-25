## Tutorial for SLTU2018

The goal of this tutorial to learn how to:
* Train and evaluate an NMT model from scratch
* Customize a hyperparameter file so you can try out new models yourself

We will use a subset of the Hindi-English data from:

```
The IIT Bombay English-Hindi Parallel Corpus
Anoop Kunchukuttan, Pratik Mehta, Pushpak Bhattacharyya
Proc. of LREC 2018
```

Schedule:
* 09:30-11:00 Lecture I: Intro to NMT
* 11:30-13:30 Lab I: Computing setup, Train 1st NMT model
* 14:30-16:00 Lab II: Train 2nd NMT model / Lecture II: Alternative NMT models
* 16:30-18:00 Lab III: Review and compare results / Lecture III: Research directions in NMT

### Step 0: Computing Setup with AWS

Please form teams of 2 or 3 people each. 
Follow instructions [here](http://kaldi-asr.org/tutorials/sltu18/sltu18.html)
to get logged in to an Amazon Web Service (AWS) instance. 
Each team will have a Linux instance with a GPU at your disposal. 

### Step 1: Quickstart on Sockeye-Recipes

First, follow the instructions [here](https://github.com/kevinduh/sockeye-recipes) to 
install sockeye-recipes in your home directory. Basically, do the following:

```bash
cd
pwd
git clone https://github.com/kevinduh/sockeye-recipes
cd sockeye-recipes
bash ./install/install_sockeye_cpu.sh
bash ./install/install_sockeye_gpu.sh
bash ./install/install_tools.sh
```

This creates conda environments for sockeye, which are activated by the sockeye-recipe scripts.
You may get some warnings like "No matching distribution found for tensorflow" and "mxnet 1.1.0 has requirement numpy<=1.13.3, but you'll have numpy 1.15.1 which is incompatible." but those can be safely ignored. 

Next, follow the tutorial in [quickstart](../../quickstart/).
This will guide you through the basics of building a model on a toy dataset (which should only take minutes). We will make sure this works, before proceeding to train on real data. 

### Step 3: Train your first NMT model

<b>Data and setup:</b> First, download the Hindi-English data into your home directory.

```bash
cd
pwd
wget http://www.cs.jhu.edu/~kevinduh/t/sltu2018/IITB-corpus.tgz
tar -xzvf IITB-corpus.tgz
```

See the files to get a feeling for the corpus: 
```bash
head -n3 IITB-corpus/IITB.*
```

We are going to train a model specified in model1.hpm. Look at this hyperparameter file and make sure you understand it: 
```bash
cat model1.hpm
```

In particular, 
* workdir=./ (we just set this to the current directory)
* modeldir=$workdir/model1 (this can be anything you like)
* rootdir=../../../ (this is where you installed sockeye-recipes. absolute path ok too)
* src=hi (we'll translate from hindi to english here.)
* trg=en (half the class should do src=hi->trg=en, and the other half src=en->trg=hi)
* train_tok=~/IITB-corpus/IITB.ted.train (the prefix specifying our tokenized train files)
* valid_tok=~/IITB-corpus/IITB.ted.dev (the prefix specifying our tokenized validation files)

<b>BPE Preprocessing:</b> Now, let's preprocess the data into BPE subword units:

```bash
bash ../../../scripts/preprocess-bpe.sh model1.hpm
```

Note we usually need to specify where to find the BPE'd data by setting `{train,valid}_bpe_{src,trg}`, but in this case, if you follow the standard settings, it will work for you. 

<b>Train:</b> We are ready to train! Again, look at the hyperparameter file model1.hpm.

The NMT model architecture hyperparameters we may be interested include:

* num_embed: size of word embeddings
* rnn_num_hidden: size of rnn hidden layer
* rnn_layers: number of stacked encoder/decoder layers
* num_words: vocabulary size of source and target

Hyperparameters for training include: 
* batch_size: sockeye-recipes default uses word batching, so this is number of words per batch
* optimizer: adam, adadelta, etc.
* initial_learning_rate: this is very important
* checkpoint_frequency: Sockeye saves a model after this many updates (which corresponds to number of minibatchs processed). Our default is 1000 is to give results sooner than later. For larger datasets, set to 4000 or so is generally reasonable. Reduce this if you want more frequent checkpoints, but note this affects two things: (a) the learning rate schedule and convergence criteria are based on number of checkpoints, so changing this may affect training speed and model at convergence. (b) if decode_and_evaluate is set to anything except 0 (meaning that we will start a separate decoder process for the validation data), having small checkpoint_frequency may mean these separate jobs get queued up--not a big deal, but slows down the overall training script. 
* max_num_epochs and max_updates: if either of these are met, training stops. We can of course stop the training process at any point and use the best model so far. Usually it is safe to set this at a high value. We set it small here to finish quicker. 
* keep_last_params: how many checkpoint models to keep. Set to -1 to keep all, but be wary of disk space

Now, let's train. 
Train the model with:
```bash
../../../scripts/train.sh -p model1.hpm -e sockeye_gpu
```

Note we are training on 1 GPU and 2 CPU slots. This is the recommended setting. 

While waiting, check these files in $modeldir: `cmdline.log` is sockeye-recipe's log and documents device information and time. `log` is sockeye's log and contains detailed training information. `metrics` contains useful statistics computed at each checkpoint. 

<b>Translate a test set and evaluate:</b> After we are finished training, we can try translating the test set: 

If you trained an Hindi->English model:
```bash
../../../scripts/translate.sh -i ~/IITB-corpus/IITB.ted.test.hi -o model1/IITB.ted.test.en.1best -p model1.hpm -e sockeye_gpu
```

Note that -i specifies the input file and -o specifies the output path.

We can compute the BLEU score by:
```bash
../../../tools/multi-bleu.perl ~/IITB-corpus/IITB.ted.test.en < model1/IITB.ted.test.en.1best
```

Congratulations! You've finished training the first NMT model

### Step 4: Train your second NMT model

Look at the results, discuss with your teammates, and think about what other things you may want to try. 

For example, make a new hyperparameter file: 
```bash
cp model1.hpm model2.hpm
```

And modify BPE, model architecture, training algorithms, etc. 
There are a range of possibilities. 

For the adventurous, you can try modifying (or making your own) ../../../scripts/train.sh 
to do CNN and Transformer architectures, which sockeye supports.
(The current tutorial in sockeye-recipes only has scripts for RNN architectures.)

Finally, compare your BLEU scores and see whether you can get improvements on this dataset!

