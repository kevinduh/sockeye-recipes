## WMT2014 English-German 

This example trains on a full English-German dataset of 4.5 million sentence pairs, drawn from WMT14 and packaged by <a href="https://nlp.stanford.edu/projects/nmt/">Stanford</a>. The results should be comparable to the <a href="https://nlp.stanford.edu/pubs/emnlp15_attn.pdf">Luong EMNLP2015 paper</a>.

In this example, we will run everything from a `$rootdir` that is not part of this egs/ source tree. First, we will copy over the script "run.sh" and hyperparameter file "model1.hpm" to the working directory of your choice (e.g. "my_working_dir"). 

```bash
cp run.sh my_working_dir/
cp model1.hpm my_working_dir/
```

Before running `run.sh`, make sure to modify `$rootdir` in the hyperparameters file to point to your sockeye-recipes directory. Also change `$workdir`. According to the model1.hpm, note the model will be trained on train.{en,de} (train_tok) and validated on newstest2013 (valid_tok).

Now we will run the entire process through one script, assuming that you are on a GPU node. See the script for invocation and modify as needed. 
```bash
run.sh
```

The downloading/preprocessing part may take up to 2 hours, and training may take up to 20 hours (on a GTX 1080 Ti GPU) for this particular hyperparameter configuration. The BLEU score on newstest2014.de after decoding should be around 19.33.
