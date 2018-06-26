## Using pre-trained embeddings with Sockeye

This is a recipe for using pre-trained embeddings with Sockeye. You may choose to initialize the Sockeye embeddings with source or target pre-trained embeddings or both.

### TLDR;
Assuming you have a pre-processed bitext (tokenized, BPE-ed) and pre-trained embeddings ready to use, copy the `example.hpm` file and change the following parameters:

```bash
embeddings_src
embeddings_trg
``` 

You will need to change other parameters to point to your dataset and your sockeye-recipes install. Also, you may choose to modify the hyperparamters. Once, you have this set up and a sockeye conda environment ready to go, use the following to start training:

```bash
$SOCKEYE_RECIPES/scripts/train-embeddings.sh -p hyperparams.txt -e ENV_NAME [-d DEVICE]
```

E.g.,

```bash
$SOCKEYE_RECIPES/scripts/train-embeddings.sh -p example.hpm -e sockeye_cpu
```

If you are working with your own pre-trained embeddings, or if you want to make custom modifications to how pre-trained embeddings are to be used, we strongly suggest that you read the detailed tutorial below.

## Detailed Tutorial

For this tutorial, we will build upon the quickstart tutorial of sockeye-recipes.

### Step 0: Download data

Let's start off by downloading a small bitext (German-English) and some pre-trained embeddings to go along with it.

```bash
./0_download_data.sh
```

This will create the `sample-de-en` folder with the some data for training and validation.
In addition, you will notice a folder named `emb`. This contains some pre-trained embeddings for German and English which we will use to initialize this NMT model.
Note: The training data is a small subset of the Europarl corpus and the pre-trained embeddings come from the TED de-en corpus (so, expect vocab mismatch).

### Step 1: Create a bitext vocab

We only really care about embeddings of words which are in our bitext. To filter and initialize embeddings, we will first extract the source and target vocabs from the bitext

```bash
python3 -m sockeye.vocab \
          -i $train_src \
          -o $modeldir/vocab.src.0.json

python3 -m sockeye.vocab \
          -i $train_trg \
          -o $modeldir/vocab.tgt.0.json
```

where `train_src` and `train_trg` are your source and target training files and `modeldir` is your model directory.

This will create the bitext vocabularies:

```bash
$modeldir/vocab.src.0.json
$modeldir/vocab.tgt.0.json
```

#### Under the hood; Pre-filtering embeddings
If you find that your pre-trained embedding vocabularies are especially large, you may choose to filter them based on the bitext vocabulary even before you start any of the initialization steps in the following sections. 

As an example, our pre-trained embeddings (the ones in `sample-de-en/emb`) came from the TED de-en corpus and had a larger vocabulary than the bitext vocabulary. They were filtered using the following steps:

```bash
$ python $SOCKEYE_RECIPES/scripts/util/create_small_emb.py \
  ~/gkumar/data/scale/pretrained_emb/fasttext/de-en/ted/train.cln.de.model.vec \
  $modeldir/vocab.src.0.json \
  sample-de-en/emb/small.cln.de.vec
Words in out (bitext) vocab : 12815
Words retained : 8962

$ python $SOCKEYE_RECIPES/scripts/util/create_small_emb.py \
  ~/gkumar/data/scale/pretrained_emb/fasttext/de-en/ted/train.cln.en.model.vec \
  $modeldir/tiny_rnn/vocab.tgt.0.json \
  sample-de-en/emb/small.cln.en.vec
Words in out (bitext) vocab : 7764
Words retained : 6571
```

### Converting embeddings into Sockeye compatible formats

First, we'll convert the pre-trained embeddings (Fasttext, in this case) to `npy` objects and their vocabularies to the JSON format so that sockeye can ingest them. The following steps will do this

```bash
$ python $SOCKEYE_RECIPES/scripts/util/vec2npy.py \
  sample-de-en/emb/small.cln.de.vec \
  sample-de-en/emb/small.cln.de.vec
Rescaled mean of emb matrix (per feature) from -0.004608003 to 8.183625e-11
Shape of emb matrix = (8962, 512)

$ python $SOCKEYE_RECIPES/scripts/util/vec2npy.py \
  sample-de-en/emb/small.cln.en.vec \
  sample-de-en/emb/small.cln.en.vec
Rescaled mean of emb matrix (per feature) from 0.0057422775 to -4.5354318e-11
Shape of emb matrix = (6571, 512)
```

This will create the necessary files:

```bash
$ ls -1 sample-de-en/emb/*.{npy,vocab}
sample-de-en/emb/small.cln.de.vec.npy
sample-de-en/emb/small.cln.de.vec.vocab
sample-de-en/emb/small.cln.en.vec.npy
sample-de-en/emb/small.cln.en.vec.vocab
```

#### Under the hood: Mean normalization of embeddings
You may have noticed a log entry from the previous step stating that the mean of the embeddings matrices was normalized.

There is some merit to making sure that the pre-trained embeddings resemble the randomly initially ones (appear to be from the same distribution). Sockeye will try and match the standard deviation of the randomly initialized embeddings to the pre-trained ones but it does not perform mean matching. To achieve this, we scale the values in the embedding matrices, per feature, so that each feature has zero mean.

```bash
Rescaled mean of emb matrix (per feature) from 0.0057422775 to -4.5354318e-11
```

### Initialize sockeye params with pre-trained values
Before starting training, we initialize the sockeye params with the pre-trained values. This will do the following: 

1. Create a randomly initialized embedding matrix for the source and target vocabs respectively.
2. For any word which has a pre-trained embedding, it's randomly initialized embedding will be replaced with the pre-trained one.
3. The size of the final embedding matrix will be `(src_vocab_size x emb_size)` and `(tgt_vocab_size x emb_size)` for the source and target languages respectively.  
4. These parameters, `source_embed_weight` and `target_embed_weight` are written to a file.

```bash
$ python3 -m sockeye.init_embedding \
  -w sample-de-en/emb/small.cln.de.vec.npy sample-de-en/emb/small.cln.en.vec.npy \
  -i sample-de-en/emb/small.cln.de.vec.vocab sample-de-en/emb/small.cln.en.vec.vocab \
  -o $modeldir/vocab.src.0.json $modeldir/vocab.tgt.0.json \
  -n source_embed_weight target_embed_weight \
  -f tiny_rnn/params.init
[INFO:__main__] Sockeye version 1.18.15 commit
[INFO:__main__] Loading input weight file: sample-de-en/emb/small.cln.de.vec.npy
[INFO:__main__] Loading input/output vocabularies: sample-de-en/emb/small.cln.de.vec.vocab tiny_rnn/vocab.src.0.json
[INFO:sockeye.vocab] Vocabulary (8962 words) loaded from "sample-de-en/emb/small.cln.de.vec.vocab"
[INFO:sockeye.vocab] Vocabulary (12815 words) loaded from "tiny_rnn/vocab.src.0.json"
[INFO:__main__] Initializing parameter: source_embed_weight
[INFO:__main__] Loading input weight file: sample-de-en/emb/small.cln.en.vec.npy
[INFO:__main__] Loading input/output vocabularies: sample-de-en/emb/small.cln.en.vec.vocab tiny_rnn/vocab.tgt.0.json
[INFO:sockeye.vocab] Vocabulary (6571 words) loaded from "sample-de-en/emb/small.cln.en.vec.vocab"
[INFO:sockeye.vocab] Vocabulary (7764 words) loaded from "tiny_rnn/vocab.tgt.0.json"
[INFO:__main__] Initializing parameter: target_embed_weight
[INFO:__main__] Saving initialized parameters to tiny_rnn/params.init
```

You should see a `$modeldir/params.init` file. We are finally ready to train.

### Training a Sockeye model with initialized parameters
Training a sockeye model should proceed as usual with the addition of the following switches:

1. `--params $modeldir/params.init` : Initializes Sockeye parameters `source_embed_weight` and `target_embed_weight` with the the values in this file.
2. `--allow-missing-params` : Allows initialization of a partial paramter set. Required for the previous step to work.
3. `--source-vocab $modeldir/vocab.src.0.json` and `--target-vocab $modeldir/vocab.tgt.0.json` : Use these pre-extracted bitext vocabs instead of running `sockeye.vocab` again.

### Source-only or target-only embeddings
This recipe supports initializing embeddings for only one language if you wish to do so. To achieve this, when using a hyperparameters file (`example.hpm)` only initialize `embeddings_src` or `embeddding_trg` depending on which side you want to initialize with pre-trained embeddings. If you specify none of these, this recipe with backoff to the default Sockeye behavior of randomly initializing word embeddings.

