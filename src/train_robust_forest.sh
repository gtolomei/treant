#!/bin/bash

if [ $# -lt 6 ]
then
    echo "Wrong number of input arguments supplied! Please, invoke the script as follows:"
    echo ""
    echo "> ./train_robust_forest.sh <DATASET_NAME> <LEARNING_ALGORITHM> <LOSS_FUNCTION> <N_ESTIMATORS> <MAX_DEPTH> <ATTACKER_BUDGET>"
    echo ""
    echo "where:"
    echo "- <DATASET_NAME> is the name of the dataset used for this training run"
    echo "- <LEARNING_ALGORITHM> is one in {standard, reduced, adv-boosting, robust, par-robust,icml2019}"
    echo "- <LOSS_FUNCTION> is one in {sse, mse, mae, gini_impurity, entropy,logloss}"
    echo "- <N_ESTIMATORS> is the number of base estimators used for training this ensemble"
    echo "- <MAX_DEPTH> is the maximum depth of the tree"
    echo "- <ATTACKER_BUDGET> is the budget of the attacker"
    echo ""
    exit 1
fi

DATASET_NAME=$1
DATASET_DIR=../data

PYTHON_SCRIPT_NAME="python3 -B ./train_robust_forest.py"

OUTPUT_ROOT_DIR=../out
MODELS_ROOT_DIR=${OUTPUT_ROOT_DIR}/models
TRAIN_SET_FILE_PATH=${DATASET_DIR}/${DATASET_NAME}/train.csv.bz2
VALID_SET_FILE_PATH=${DATASET_DIR}/${DATASET_NAME}/valid.csv.bz2
TEST_SET_FILE_PATH=${DATASET_DIR}/${DATASET_NAME}/test.csv.bz2
MODELS_DIR=${MODELS_ROOT_DIR}/${DATASET_NAME}

LEARNING_ALGORITHM=$2
LOSS_FUNCTION=$3
N_ESTIMATORS=$4
MAX_DEPTH=$5
ATTACKER_BUDGET=$6
ATTACKS_FILE=${DATASET_DIR}/${DATASET_NAME}/attacks/${DATASET_NAME}
ATTACK_RULES_FILE=${DATASET_DIR}/${DATASET_NAME}/attacks/attacks.json

# HYPERPARAMETERS
N_INSTANCES=30000
N_INSTANCES_PER_NODE=20
FEATURE_SAMPLING=80
INSTANCE_SAMPLING=80
JOBS=40

# Check if there is any log file in the current working directory
count=`ls -1 ./train_robust_forest_${DATASET_NAME}.log 2>/dev/null | wc -l`
if [ $count != 0 ]
then 
    echo "==> Cleaning up log files..."
    echo "rm ./train_robust_forest_${DATASET_NAME}.log"
    rm ./train_robust_forest_${DATASET_NAME}.log
fi 


CMD_NAME="${PYTHON_SCRIPT_NAME} $DATASET_NAME $TRAIN_SET_FILE_PATH $VALID_SET_FILE_PATH $TEST_SET_FILE_PATH $LEARNING_ALGORITHM $N_ESTIMATORS -l $LOSS_FUNCTION -o $MODELS_DIR -n $N_INSTANCES -d $MAX_DEPTH -i $N_INSTANCES_PER_NODE -b $ATTACKER_BUDGET -a $ATTACKS_FILE -r $ATTACK_RULES_FILE -bf $FEATURE_SAMPLING -bs $INSTANCE_SAMPLING --jobs $JOBS"

echo "==> Training $LEARNING_ALGORITHM model..."
echo $CMD_NAME
$CMD_NAME
