#!/bin/sh

# The Multitarget TED talks dataset has many languages
# This script sets things up for a particular language-pair
# Specify the source language (src) and it will create 
# hyperparameter files, etc. for that src-en task. 

if [ $# -ne 1 ]; then
    echo "Usage: 1_setup_task.sh src"
    echo "  where src is a language identifier {zh, ko, fa, ..}"
fi

src=$1 # e.g. zh

mkdir -p ${src}-en/data-bpe
for m in model1 ; do
    sed "s#__SRC__#${src}#g" $m.hpm-template > ${src}-en/$m.hpm
done
