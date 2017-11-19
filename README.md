# sockeye-recipes
Training scripts and recipes for the Sockeye Neural Machine Translation (NMT) toolkit
- The original Sockeye codebase is at [AWS Labs](https://github.com/awslabs/sockeye)
- This version is based off [a fork](https://github.com/kevinduh/sockeye)

This package contains scripts that makes it easy to run NMT experiments.
All settings are specified in a user file like "hyperparams.txt", and used by:
- scripts/preprocess-bpe.sh: Preprocess bitext via BPE segmentation
- scripts/train.sh: Train the NMT model given bitext
- scripts/plot-validation-curve.sh: Compute BLEU curves by iteration on validation data

## Installation
We assume that Anaconda for Python virtual environments is available on the system.
Run the following to install Sockeye in two Anaconda environments, sockeye_cpu and sockeye_gpu: 

```bash
> sh ./install/install_sockeye_cpu.sh
> sh ./install/install_sockeye_gpu.sh
```

The training scripts and recipes will activate either the sockeye_cpu or sockeye_gpu environment depending on whether CPU or GPU is specified. 
