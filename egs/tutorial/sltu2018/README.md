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
Decide a TeamName and decide whether to work on Hindi-to-English or English-to-Hindi translation. Sign up on this [Google sheet](https://docs.google.com/spreadsheets/d/1nQDbDsY_LlkIEvkrJ0-niqpH8aZW3BS5E-pT7f_jMLs/edit?usp=sharing).


Follow these instructions to get logged in to an 
Amazon Web Service (AWS) instance:

<b>GNU Linux, Unix, MacOS X</b>
Using the provided [SSH key](http://www.cs.jhu.edu/~kevinduh/t/sltu2018/sltu18_public.pem), login on an assigned machine using the username ec2-user. Using the command line SSH, this can be done using the following command:
   
```bash
ssh -i sltu18_public.pem ec2-user@<machine-name>
```

where <machine-name> is the address of the machine with the same number you've been assigned. Please note that after downloading, you might need to change the access rights to the file sltu18_public.pem to allow only the current user to read the file (it's a security precaution built-in into OpenSSH clients). This can be done using the following command

```bash
chmod 600 sltu18_public.pem
```

<b>MS Windows</b>
For MS Windows, no SSH client is comming as a standard part of the OS. We suggest to install [PuTTY](http://www.chiark.greenend.org.uk/~sgtatham/putty/). We provide additional information about PuTTY setup [here](http://kaldi-asr.org/tutorials/sltu18/sltu18-putty.html). Please note that PuTTY needs the key in a special format. You can download the key [here](http://www.cs.jhu.edu/~kevinduh/t/sltu2018/sltu18_public.ppk).

Once you've log in, run the command `screen -S YourTeamName` (or tmux if you prefer). That command ensures that even if the connection to the machine is dropped, the scripts will still keep running, so you won't have to run everything from scratch.

In case you need to restore the connection (for example after a lunch break), use the command screen -ls to see the already opened sessions. Also, it is possible to see two (or even more) sessions, especially in case you were working in pairs (but each on their own computer). You can guess which is the right by looking at the date of the session and it's state. You want to primarily consider the sessions that are in "Dettached" state.

Important: after you're done, set the following according to your assigned device on the grid:

```bash
export CUDA_VISIBLE_DEVICES=$YOUR_ASSIGNED_DEVICE
```

### Step 1: Quickstart on Sockeye-Recipes

Different teams will be sharing the same AWS instance with the same ec2-user login. To prevent conflicts, everyone should do their work under a TeamName directory created as such: 

```bash
mkdir ~/YourTeamName
cd ~/YourTeamName
pwd
```

Install sockeye-recipes:

```
git clone https://github.com/kevinduh/sockeye-recipes
cd sockeye-recipes
bash ./install/install_tools.sh
```

Note, if you are doing this from scratch by yourself on your own machine, you would also need to install the core sockeye engine via `./install/install_sockeye_cpu.sh` and `./install/install_sockeye_gpu.sh`. These scripts create conda environments for sockeye, which are activated by the sockeye-recipe scripts. You may get some warnings like "No matching distribution found for tensorflow" and "mxnet 1.1.0 has requirement numpy<=1.13.3, but you'll have numpy 1.15.1 which is incompatible." but those can be safely ignored. In any case, this has already been done for you this time, so can be skipped.

Next, follow the tutorial in [quickstart](../../quickstart/).
This will guide you through the basics of building a model on a toy dataset (which should only take minutes). We will make sure this works, before proceeding to train on real data. 

### Step 3: Train your first NMT model

<b>Data and setup:</b> First, the Hindi-English data is already downloaded 
but if you need it for your own purposes, get it from [here](http://www.cs.jhu.edu/~kevinduh/t/sltu2018/IITB-corpus.tgz)

See the files to get a feeling for the corpus: 
```bash
head -n3 IITB-corpus/IITB.*
```

We are going to train a model specified in model1.hpm. Look at this hyperparameter file and make sure you understand it: 
```bash
cd YourTeamName/sockeye-recipes/egs/tutorial/sltu2018/
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
cd YourTeamName/sockeye-recipes/egs/tutorial/sltu2018/
bash ../../../scripts/preprocess-bpe.sh model1.hpm
```

Checkout the results in `data-bpe`.
Note we usually need to specify where to find the BPE'd data by setting `{train,valid}_bpe_{src,trg}`, but in this case, if you follow the standard settings, it will work for you. 

<b>Train:</b> We are ready to train! Before doing so, make sure you are in a tmux or screen session so your long job won't get lost. 

Again, let's look one more time at the hyperparameter file model1.hpm. The NMT model architecture hyperparameters we may be interested include:

* num_embed: size of word embeddings
* rnn_num_hidden: size of rnn hidden layer
* rnn_layers: number of stacked encoder/decoder layers
* num_words: vocabulary size of source and target

Hyperparameters for training include: 
* batch_size: sockeye-recipes default uses word batching, so this is number of words per batch
* optimizer: adam, adadelta, etc.
* initial_learning_rate: this is very important
* checkpoint_frequency: Sockeye saves a model after this many updates (which corresponds to number of minibatchs processed). Here we set it small to give results sooner than later. For larger datasets, set to 4000 or so is generally reasonable. Reduce this if you want more frequent checkpoints, but note this affects two things: (a) the learning rate schedule and convergence criteria are based on number of checkpoints, so changing this may affect training speed and model at convergence. (b) if decode_and_evaluate is set to anything except 0 (meaning that we will start a separate decoder process for the validation data), having small checkpoint_frequency may mean these separate jobs get queued up--not a big deal, but slows down the overall training script. 
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

There are a range of possibilities, including:
- Try BPE preprocessing, use different number of operations
- Try different model architecture (e.g. num_embed, num_hidden) or regularization (dropout rate)
- Try different training algorithm (e.g. sgd, adadelta, adagrad, eve)
- Try training on model1 longer
- etc.

For the adventurous, you can try modifying (or making your own) ../../../scripts/train.sh 
to do CNN and Transformer architectures, which sockeye supports.
(The current tutorial in sockeye-recipes only has scripts for RNN architectures.)

Finally, compare your BLEU scores and see whether you can get improvements on this dataset!

