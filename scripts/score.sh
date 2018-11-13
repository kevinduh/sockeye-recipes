#!/bin/bash
#
# Score some existing translations with a Sockeye NMT model

function errcho() {
  >&2 echo $1
}

function show_help() {
  errcho "Usage: score.sh -p hyperparams.txt -s source -t target -o output -e ENV_NAME [-d DEVICE]"
  errcho "Source is a source text file to be translated"
  errcho "Target is filename for target translations"
  errcho "Output is the filename where scores for the translations will be stored"
  errcho "ENV_NAME is the sockeye conda environment name"
  errcho "Device is optional and inferred from ENV"
  errcho ""
}

function check_file_exists() {
  if [ ! -f $1 ]; then
    errcho "FATAL: Could not find file $1"
    exit 1
  fi
}

while getopts ":h?p:e:s:t:o:d:" opt; do
  case "$opt" in
    h|\?)
      show_help
      exit 0
      ;;
    p) HYP_FILE=$OPTARG
      ;;
    e) ENV_NAME=$OPTARG
      ;;
    s) SOURCE_FILE=$OPTARG
      ;;
    t) TARGET_FILE=$OPTARG
      ;;
    o) OUTPUT_FILE=$OPTARG
       ;;
    d) DEVICE=$OPTARG
      ;;
  esac
done

if [[ -z $HYP_FILE || -z $ENV_NAME || -z $SOURCE_FILE || -z $TARGET_FILE || -z $OUTPUT_FILE ]]; then
    errcho "Missing arguments"
    show_help
    exit 1
fi

###########################################
# (0) Setup
# source hyperparams.txt to get text files and all training hyperparameters
check_file_exists $HYP_FILE
check_file_exists $SOURCE_FILE
check_file_exists $TARGET_FILE
source $HYP_FILE
source activate $ENV_NAME

# options for cpu vs gpu training (may need to modify for different grids)
source $rootdir/scripts/get-device.sh $DEVICE ""

###########################################
# (1) Book-keeping
LOG_FILE=${OUTPUT_FILE}.log
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "Start scoring: $datenow on $(hostname)" > $LOG_FILE
echo "$0 $@" >> $LOG_FILE
echo "$devicelog" >> $LOG_FILE

###########################################
# (2) Score!
python -m sockeye.score -m $modeldir --source $SOURCE_FILE --target $TARGET_FILE $device \
--disable-device-locking \
$CHECKPOINT \
--max-seq-len $max_seq_len 2>> $LOG_FILE > $OUTPUT_FILE

##########################################
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "End scoring: $datenow on $(hostname)" >> $LOG_FILE
echo "===========================================" >> $LOG_FILE
