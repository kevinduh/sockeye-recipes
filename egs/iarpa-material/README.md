## NMT Models for IARPA MATERIAL datasets

This recipe builds systems from the IARPA MATERIAL datasets.
Hyperparameter (hpm) files with well-tuned values are provided here. 

If you want to [train models from scratch](#scratch), you will need to point the hpm files to your own copy of the data. 

If you want to [decode with existing trained models](#existing), they are available at: 
[http://www.cs.jhu.edu/~kevinduh/a/sockeye-recipes-models/egs/iarpa-material](http://www.cs.jhu.edu/~kevinduh/a/sockeye-recipes-models/egs/iarpa-material)



### Summary

This recipes contains well-tuned models for low-resource scenarios. 
We provide Somali(so)-to-English(en) and Swahili(sw)-to-English(en) systems based on the IARPA MATERIAL datasets. 

We have five training data conditions. The baseline condition uses the MATERIAL Build Packs (24k parallel sentences for both Somali and Swahili). We then add Dictionary (e.g. Panlex), Paracrawl, and other Found-Bitext (e.g. Tanzil, Global Voices). Fo tuning/validation, we used MATERIAL's ANALYSIS2-text data; for testing, we used the larger ANALYSIS1-text data (among others). 

The data size in terms of number of parallel sentences, and the Test BLEU scores for different training conditions are summarized below:

| Training condition                      | Size | BLEU  | 
| :-------------------------------------- | :--- | :---- |
| Somali-English baseline                 | 24k  | 14.4  |
| + paracrawl                             | 104k | 20.2  |
| + dictionary                            | 50k  | 14.3  | 
| + dictionary + found-bitext             | 273k | 24.4  |
| + dictionary + found-bitext + paracrawl | 354k | 25.0  | 

| Training condition                      | Size | BLEU  | 
| :-------------------------------------- | :--- | :---- |
| Swahili-English baseline                | 24k  | 24.8  |
| + paracrawl                             | 85k  | 26.6  |
| + dictionary                            | 123k | 25.3  | 
| + dictionary + found-bitext             | 312k | 33.3  |
| + dictionary + found-bitext + paracrawl | 373k | 33.7  | 

For low-resource training conditions, we found that it is very important (even more so than high-resource conditions) to carefully tune the hyperparameters of the model. The models in this recipe are the result of extensive hyperparameter optimization. Our hyperparameter optimization focused on transformer models and searched over the following ranges:

| Hyperparameter      | Possible Settings | 
| :------------------ | :---------------- |
| BPE symbols         | 1000, 2000* , 4000, 8000, 16000* , 32000*  |
| Number of embedding | 256, 512, 1024*          |
| Number of layers    | 1* , 2, 4              |
| Initial learning rate | 0.0003, 0.0006, 0.001 |
| Transformer model size | 256, 512, 1024* | 
| Attention heads | 8, 16 |
| Transformer FF Hidden size | 1024, 2048* | 

(For the baseline condition, we ran an even larger hyperparameter search that additionally included the settings labeled by an asterisk * ) 

<a name="scratch"></a>
### To replicate results and train from scratch

You should see files like `baseline.hpm` and `baseline+dictionary.hpm` in the subdirectories `so-en` or `sw-en`. These represent the hyperparameter files found by the hyperparameter optimization for each of the training conditions above. 

Pick the one you want to train. We will give an example for `so-en/baseline.hpm` here. 

First, modify the variables in `so-en/baseline.hpm` to adapt to the paths in your system. In particular:

* Set `workdir` and `modeldir` to the location you want to save the model. 
* Set `rootdir` to your copy of sockeye-recipes
* Set either `origdata`, or `train_tok` and `valid_tok` to point to your training data. In this case, point to the MATERIAL Build Pack. Note that sockeye-recipes assumes tokenized text; we used Joshua's standard tokenization in these experiments.

You can check out the other hyperparameters in the hpm file to see what kind of model will be trained. 

Now, we are ready to run BPE preprocessing on the train and validation data: 

```bash
../../scripts/preprocess-bpe.sh so-en/baseline.hpm
```

The resulting BPE vocabulary file (for English and Somali) will be in: `$workdir/data-bpe/train.*.bpe_vocab` and the segmented training file is: `$workdir/data-bpe/train.bpe-*.{so,en}`. 

To train, we will use qsub and gpu (modify this command for your own computer system:

```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=12:00:00,num_proc=2 -j y ../../scripts/train.sh -p so-en/baseline.hpm -e sockeye_gpu
```

After training, we can decode some test set with:

```base
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=00:30:00 -j y ../../scripts/translate.sh -p so-en/baseline.hpm -i INPUT -o OUTPUT -e sockeye_gpu
```

<a name="existing"></a>
### To use the existing trained models

First, download the model you want from [http://www.cs.jhu.edu/~kevinduh/a/sockeye-recipes-models/egs/iarpa-material](http://www.cs.jhu.edu/~kevinduh/a/sockeye-recipes-models/egs/iarpa-material).

Assuming you have sockeye-recipes installed, these models should contain all the needed parameters to decode some test data. 

As an example, suppose we download and unpack [Swahili baseline+paracrawl model](http://www.cs.jhu.edu/~kevinduh/a/sockeye-recipes-models/egs/iarpa-material/sw-en/baseline+paracrawl.tgz).
To use this model in your system, first make sure that you have sockeye-recipes installed. Then we need to set these variables in `sw-en/baseline+paracrawl.hpm`:

* Set `modeldir` to the location of the directory of the unpacked downloaded model. 
* Set `rootdir` to your copy of sockeye-recipes
* Change `train_bpe_src` to `train_bpe_src=$modeldir/train.bpe-${bpe_symbols_src}.$src`. This points to the bpe_vocab file included in the download. 

Finally, you should be able to translate!

```base
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=00:30:00 -j y ../../scripts/translate.sh -p sw-en/baseline+paracrawl.hpm -i INPUT -o OUTPUT -e sockeye_gpu
```




