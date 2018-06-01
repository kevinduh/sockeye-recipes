#!/bin/bash

# This script sets up curriculum learning for TED de-en

src=de
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

    echo "" >> ${src}-${trg}/$model.hpm
    echo "# For curriculum learning" >> ${src}-${trg}/$model.hpm
    echo "score_file=\${workdir}/curriculum_sent.scores" >> ${src}-${trg}/$model.hpm
    echo "curriculum_update_freq=1000" >> ${src}-${trg}/$model.hpm

done
