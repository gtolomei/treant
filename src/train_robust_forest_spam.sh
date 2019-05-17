#!/bin/bash

if [ $# -lt 4 ]
then
    echo "Wrong number of input arguments supplied! Please, invoke the script as follows:"
    echo ""
    echo "> ./train_robust_forest_spam.sh <LEARNING_ALGORITHM> <N_ESTIMATORS> <MAX_DEPTH> <ATTACKER_BUDGET>"
    echo ""
    echo "where:"
    echo "- <LEARNING_ALGORITHM> is one in {standard, reduced, adv-boosting, robust}"
    echo "- <N_ESTIMATORS> is the number of base estimators used for training this ensemble"
    echo "- <MAX_DEPTH> is the maximum depth of the tree"
    echo "- <ATTACKER_BUDGET> is the budget of the attacker"
    echo ""
    exit 1
fi

DATASET_NAME=spam
DATASET_DIR=../data

PYTHON_SCRIPT_NAME="python3 -B ./train_robust_forest_${DATASET_NAME}.py"

OUTPUT_ROOT_DIR=../out
MODELS_ROOT_DIR=${OUTPUT_ROOT_DIR}/models
TRAIN_SET_FILE_PATH=${DATASET_DIR}/${DATASET_NAME}/train.csv.bz2
VALID_SET_FILE_PATH=${DATASET_DIR}/${DATASET_NAME}/valid.csv.bz2
TEST_SET_FILE_PATH=${DATASET_DIR}/${DATASET_NAME}/test.csv.bz2
MODELS_DIR=${MODELS_ROOT_DIR}/${DATASET_NAME}

LEARNING_ALGORITHM=$1
N_ESTIMATORS=$2
MAX_DEPTH=$3
ATTACKER_BUDGET=$4
ATTACKS_FILE=${DATASET_DIR}/${DATASET_NAME}/attacks/${DATASET_NAME}

# HYPERPARAMETERS
N_INSTANCES=1000
N_INSTANCES_PER_NODE=20

# Check if there is any log file in the current working directory
count=`ls -1 ./train_robust_forest_${DATASET_NAME}.log 2>/dev/null | wc -l`
if [ $count != 0 ]
then 
    echo "==> Cleaning up log files..."
    echo "rm ./train_robust_forest_${DATASET_NAME}.log"
    rm ./train_robust_forest_${DATASET_NAME}.log
fi 


CMD_NAME="${PYTHON_SCRIPT_NAME} $TRAIN_SET_FILE_PATH $VALID_SET_FILE_PATH $TEST_SET_FILE_PATH $LEARNING_ALGORITHM $N_ESTIMATORS -l sse -o $MODELS_DIR -n $N_INSTANCES -d $MAX_DEPTH -i $N_INSTANCES_PER_NODE -b $ATTACKER_BUDGET -a $ATTACKS_FILE"

echo "==> Training $LEARNING_ALGORITHM model..."

if [ "$LEARNING_ALGORITHM" = "reduced" ] 
then
    CMD_NAME="$CMD_NAME -xf workclass marital_status occupation education_num capital_gain hours_per_week"
fi

if [ $N_ESTIMATORS -gt 1 -a "$LEARNING_ALGORITHM" != "adv-boosting" ] 
then
    CMD_NAME="$CMD_NAME -bs 95 -bf 25"
fi


echo $CMD_NAME
$CMD_NAME