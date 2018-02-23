# Auto-tuning

Scripts for automatically tuning hyperparameters of Sockeye Neural Machine Translation (NMT) models.

- This code applies the covariance matrix adaption-evolution strategy(CMA-ES) and a Pareto-based multi-objective CMA-ES to generate better values for hyperparameters and optimize multiple objects, e.g. translation performance and computational time. It is based on the paper "[Evolution Strategy Based Automatic Tuning of
Neural Machine Translation Systems](http://cs.jhu.edu/~kevinduh/papers/qin17evolution.pdf)"[Hao Qin, Takahiro Shinozaki, Kevin Duh].

- This code is a modification version of [Hao Qin & Bairong Zhang's code](https://github.com/marvinzh/cma_es)(Shinozaki-lab, Tokyo Institute of Technology).

## Usage

(1) Prerequisite

```bash
pip install cma 
```

(2) Configuration

``hyperparameters.txt``: The trainig and auto-tuning will use the settings specified in this configuration file.

- Configure working and data directories.

- Configure Sockeye NMT training parameters.

- Configure auto-tuning settings: initial values and mapping functions(hyperparameters need to be mapped to the same scale) for hyperparameters, generation and population numbers, saving directories, etc..

``parallel.py``: If you plan to use GPUs to train your NMT models, you need to also configure the command for submiiting gpu tasks.

(3) Run

- GPU version

```bash
sh ~/path/to/sockeye-recipies/auto-tuning/auto-tune.sh ~/path/to/sockeye-recipies/auto-tuning/hyperparams.txt gpu n
```

``n`` is the number of devices used for auto-tuning.

- CPU version

```bash
sh ~/path/to/sockeye-recipies/auto-tuning/auto-tune.sh ~/path/to/sockeye-recipies/auto-tuning/hyperparams.txt cpu n
```

## Results 

During training, some files will be generated.

``checkpoints``: Records genes of the latest generation for the use of ``cma``'s ``ask`` and ``tell``. 

``generation_00/``: This folder contains all models and gene information for the first generation.

``generation_00/genes/``: This folder contains hyperparameter settings(genes) for each model trained in the first generation.

``generation_00/genes.src``: Records the score of each model in the first generation. The default measurment is BLEU.

``generation_00/model_00/``: Information of the first model of the first generation.


