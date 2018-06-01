#!/bin/bash

# The Multitarget TED talks dataset has many languages
# This script sets things up for a particular language-pair
# Specify the source language (src) and it will create 
# hyperparameter files, etc. for that src-en task. 

if [ $# -ne 1 ]; then
    echo "Usage: 1_setup_task.sh src"
    echo "  where src is a language identifier {zh, ko, fa, ..}"
fi

src=$1 # e.g. zh
trg=en

rootdir="$(readlink -f "$(dirname "$0")/../../")"
train_tok="$rootdir/egs/ted/multitarget-ted/en-${src}/tok/ted_train_en-${src}.tok.clean"
valid_tok="$rootdir/egs/ted/multitarget-ted/en-${src}/tok/ted_dev_en-${src}.tok"
workdir="./"

mkdir -p ${src}-${trg}

for model in rs1 ; do
    sed "s#__SRC__#${src}#g; s#__TRG__#${trg}#g;" $rootdir/hpm/$model.hpm-template \
	| sed "s#__WORKDIR__#${workdir}#g; " \
	| sed "s#__ROOTDIR__#${rootdir}#g; " \
	| sed "s#__TRAIN_TOK__#${train_tok}#g; " \
	| sed "s#__VALID_TOK__#${valid_tok}#g; " > ${src}-${trg}/$model.hpm
done
