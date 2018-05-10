#!/bin/bash
#
# Preprocess train and validation data with BPE

if [ $# -ne 1 ]; then
    echo "Usage: preprocess-bpe.sh hyperparams.txt"
    exit
fi


###########################################
# (0) Hyperparameter settings 
## source hyperparams.txt to get text files and #symbols for BPE
source $1

## standard settings, need not modify
bpe_minfreq=2 


# (1) Save new BPE'ed data and vocab file in datadir 
subword=$rootdir/tools/subword-nmt/
mkdir -p $datadir


###########################################
# (2) BPE on source side
echo `date '+%Y-%m-%d %H:%M:%S'` "- Learning BPE on source and creating vocabulary: $bpe_vocab_src"
python $subword/learn_bpe.py --input ${train_tok}.$src --output $bpe_vocab_src --symbols $bpe_symbols_src --min-frequency $bpe_minfreq 

echo `date '+%Y-%m-%d %H:%M:%S'` "- Applying BPE, creating: ${train_bpe}.$src, ${valid_bpe}.$src" 
python $subword/apply_bpe.py --input ${train_tok}.$src --codes $bpe_vocab_src --output $train_bpe_src
python $subword/apply_bpe.py --input ${valid_tok}.$src --codes $bpe_vocab_src --output $valid_bpe_src


###########################################
# (3) BPE on target side
echo `date '+%Y-%m-%d %H:%M:%S'` "- Learning BPE on target and creating vocabulary: $bpe_vocab_trg"
python $subword/learn_bpe.py --input ${train_tok}.$trg --output $bpe_vocab_trg --symbols $bpe_symbols_trg --min-frequency $bpe_minfreq 

echo `date '+%Y-%m-%d %H:%M:%S'` "- Applying BPE, creating: ${train_bpe}.$trg, ${valid_bpe}.$trg" 
python $subword/apply_bpe.py --input ${train_tok}.$trg --codes $bpe_vocab_trg --output $train_bpe_trg
python $subword/apply_bpe.py --input ${valid_tok}.$trg --codes $bpe_vocab_trg --output $valid_bpe_trg

echo `date '+%Y-%m-%d %H:%M:%S'` "- Done with preprocess-bpe.sh"
