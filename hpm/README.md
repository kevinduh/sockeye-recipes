## Hyperparameter file templates for sockeye-recipes

These files should be self-explanatory. The general naming convention is:

* First letter: {r = RNN, c = CNN, t = Transformer}
* Second letter: {s = small model, m = mid-sized model, l = large model}
* Third numeral: Just an identifier

So, for example:

* rs1 = RNN-based seq2seq model, relatively small size, id=1
* rs2 = yet another small RNN seq2seq model
* cm1 = CNN-based model, mid-sized

These hyperparameter file templates are meant to be used as starting points. The best hyperparameter setting will of course depend on the task. 

