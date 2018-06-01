#!/bin/bash
#
# Translate an input file with a Sockeye NMT model

function errcho() {
  >&2 echo $1
}

function show_help() {
  errcho "Usage: translate.sh -p hyperparams.txt -i input -o output -e ENV_NAME [-d DEVICE]"
  errcho "input is a text file to be translated"
  errcho "output is path of generated translations"
  errcho "Device is optional and inferred from env"
  errcho ""
}

function check_file_exists() {
  if [ ! -f $1 ]; then
    errcho "FATAL: Could not find file $1"
    exit 1
  fi
}

while getopts ":h?p:e:i:o:d:" opt; do
  case "$opt" in
    h|\?)
      show_help
      exit 0
      ;;
    p) HYP_FILE=$OPTARG
      ;;
    e) ENV_NAME=$OPTARG
      ;;
    i) INPUT_FILE=$OPTARG
      ;;
    o) OUTPUT_FILE=$OPTARG
      ;;
    d) DEVICE=$OPTARG
      ;;
  esac
done


###########################################
# (0) Setup
# source hyperparams.txt to get text files and all training hyperparameters
check_file_exists $HYP_FILE
check_file_exists $INPUT_FILE
source $HYP_FILE
source activate $ENV_NAME

# options for cpu vs gpu training (may need to modify for different grids)
source $rootdir/scripts/get-device.sh $DEVICE ""

###########################################
# (1) Book-keeping
LOG_FILE=${OUTPUT_FILE}.log
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "Start translating: $datenow on $(hostname)" > $LOG_FILE
echo "$0 $@" >> $LOG_FILE
echo "$devicelog" >> $LOG_FILE

###########################################
# (2) Translate!
subword=$rootdir/tools/subword-nmt/
### Apply BPE to input, run Sockeye.translate, then de-BPE ###
python $subword/apply_bpe.py --input $INPUT_FILE --codes $bpe_vocab_src | \
    python -m sockeye.translate --models $modeldir $device \
    --disable-device-locking \
    --max-input-len 100 2>> $LOG_FILE | \
    sed -r 's/@@( |$)//g' > $OUTPUT_FILE 


##########################################
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "End translating: $datenow on $(hostname)" >> $LOG_FILE
echo "===========================================" >> $LOG_FILE
