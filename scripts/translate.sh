#!/bin/bash
#
# Translate an input file with a Sockeye NMT model

function errcho() {
  >&2 echo $1
}

function show_help() {
  errcho "Usage: translate.sh -p hyperparams.txt -i input -o output -e ENV_NAME [-d DEVICE] [-c checkpoint] [-s]"
  errcho "Input is a source text file to be translated"
  errcho "Output is filename for target translations"
  errcho "ENV_NAME is the sockeye conda environment name"
  errcho "Device is optional and inferred from ENV"
  errcho "Checkpoint is optional and specifies which model checkpoint to use while decoding (-c 00005)"
  errcho "-s is optional and skips BPE processing on input source"
  errcho ""
}

function check_file_exists() {
  if [ ! -f $1 ]; then
    errcho "FATAL: Could not find file $1"
    exit 1
  fi
}

while getopts ":h?p:e:i:o:d:c:s" opt; do
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
    c) CHECKPOINT=$OPTARG
      ;;
    s) SKIP_SRC_BPE=1
      ;;
  esac
done

if [[ -z $HYP_FILE || -z $ENV_NAME || -z $INPUT_FILE || -z $OUTPUT_FILE ]]; then
    errcho "Missing arguments"
    show_help
    exit 1
fi

###########################################
# (0) Setup
# source hyperparams.txt to get text files and all training hyperparameters
check_file_exists $HYP_FILE
check_file_exists $INPUT_FILE
source $HYP_FILE
source activate $ENV_NAME

# options for cpu vs gpu training (may need to modify for different grids)
source $rootdir/scripts/get-device.sh $DEVICE ""

# If the checkpoint is provided, add the argument tag
[ -z $CHECKPOINT ] || CHECKPOINT="-c $CHECKPOINT"

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
max_input_len=100

if [ "$SKIP_SRC_BPE" == 1 ]; then
    ### Run Sockeye.translate, then de-BPE:
    echo "Directly translating source input without applying BPE" >> $LOG_FILE
    cat $INPUT_FILE | \
	python -m sockeye.translate --models $modeldir $device \
	--disable-device-locking \
  $CHECKPOINT \
	--max-input-len $max_input_len 2>> $LOG_FILE | \
	sed -r 's/@@( |$)//g' > $OUTPUT_FILE 
else
    ### Apply BPE to input, run Sockeye.translate, then de-BPE ###
    echo "Apply BPE to source input" >> $LOG_FILE
    python $subword/apply_bpe.py --input $INPUT_FILE --codes $bpe_vocab_src | \
	python -m sockeye.translate --models $modeldir $device \
	--disable-device-locking \
  $CHECKPOINT \
	--max-input-len $max_input_len 2>> $LOG_FILE | \
	sed -r 's/@@( |$)//g' > $OUTPUT_FILE 
fi

##########################################
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "End translating: $datenow on $(hostname)" >> $LOG_FILE
echo "===========================================" >> $LOG_FILE
