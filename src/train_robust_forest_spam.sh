#!/bin/bash

if [ $# -lt 4 ]
then
    echo "Wrong number of input arguments supplied! Please, invoke the script as follows:"
    echo ""
    echo "> ./train_robust_forest.sh <LEARNING_ALGORITHM> <N_ESTIMATORS> <MAX_DEPTH> <ATTACKER_BUDGET>"
    echo ""
    echo "where:"
    echo "- <LEARNING_ALGORITHM> is one in {standard, reduced, adv-boosting, robust}"
    echo "- <N_ESTIMATORS> is the number of base estimators used for training this ensemble"
    echo "- <MAX_DEPTH> is the maximum depth of the tree"
    echo "- <ATTACKER_BUDGET> is the budget of the attacker"
    echo ""
    exit 1
fi

PYTHON_SCRIPT_NAME="python3 -B ./train_robust_forest.py"
DATASET_NAME=census
DATASET_DIR=../data
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
count=`ls -1 ./train_robust_forest.log 2>/dev/null | wc -l`
if [ $count != 0 ]
then 
    echo "==> Cleaning up log files..."
    echo "rm ./train_robust_forest.log"
    rm ./train_robust_forest.log
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

#######################################################################################
#
# 1. TRAINING UNATTACKED SINGLE DECISION TREE
#   rdt = RobustDecisionTree(0, 
#                            attacker=EMPTY_ATTACKER,
#                            split_optimizer=OPTIMIZER, 
#                            max_depth=MAX_DEPTH, 
#                            min_instances_per_node=N_INSTANCES_PER_NODE, 
#                            max_samples=1.0,
#                            max_features=1.0,
#                            feature_blacklist=set([]),
#                           )
#   rf = RobustForest(0,
#                     base_estimator=rdt,
#                     n_estimators=1
#                    )
#
# ./${PYTHON_SCRIPT_NAME} $TRAIN_SET_FILE_PATH $VALID_SET_FILE_PATH $TEST_SET_FILE_PATH 
#                         $STD_MODEL 1 -l sse -o $MODELS_DIR -n 100 -d 8 -i 32
#                         -b 0 -a ../out/census_attacks
#
#######################################################################################
#
# 2. TRAINING ATTACKED SINGLE DECISION TREE
#   rdt = RobustDecisionTree(0, 
#                            attacker=ATTACKER,
#                            split_optimizer=OPTIMIZER, 
#                            max_depth=MAX_DEPTH, 
#                            min_instances_per_node=N_INSTANCES_PER_NODE, 
#                            max_samples=1.0,
#                            max_features=1.0,
#                            feature_blacklist=set([]),
#                           )
#   rf = RobustForest(0,
#                     base_estimator=rdt,
#                     n_estimators=1
#                    )
#
# ./${PYTHON_SCRIPT_NAME} $TRAIN_SET_FILE_PATH $VALID_SET_FILE_PATH $TEST_SET_FILE_PATH 
#                         $ROBUST_MODEL 1 -l sse -o $MODELS_DIR -n 100 -d 8 -i 32
#                         -b 60 -a ../out/census_attacks
#
#######################################################################################
#
# 3. TRAINING UNATTACKED REDUCED RANDOM FOREST
#   rdt = RobustDecisionTree(0, 
#                            attacker=EMPTY_ATTACKER,
#                            split_optimizer=OPTIMIZER, 
#                            max_depth=MAX_DEPTH, 
#                            min_instances_per_node=N_INSTANCES_PER_NODE, 
#                            max_samples=1.0,
#                            max_features=0.25,
#                            feature_blacklist={workclass, marital_status, occupation, 
#                                               education_num, capital_gain, hours_per_week},
#                           )
#   rf = RobustForest(0,
#                     base_estimator=rdt,
#                     n_estimators=1000
#                    )
#
# ./${PYTHON_SCRIPT_NAME} $TRAIN_SET_FILE_PATH $VALID_SET_FILE_PATH $TEST_SET_FILE_PATH 
#                         $REDUCED_MODEL 1000 -l sse -o $MODELS_DIR -n 100 -d 8 -i 32
#                         -b 0 -a ../out/census_attacks
#
#
#######################################################################################
#
# 4. TRAINING ATTACKED ADVERSARIAL BOOSTING
#   rdt = RobustDecisionTree(0, 
#                            attacker=ATTACKER, // NON AFFINE!!!!!!!
#                            split_optimizer=OPTIMIZER, 
#                            max_depth=MAX_DEPTH, 
#                            min_instances_per_node=N_INSTANCES_PER_NODE, 
#                            max_samples=1.0,
#                            max_features=1.0,
#                            feature_blacklist=set([]),
#                           )
#   ab = AdversarialBoosting(0,
#                            base_estimator=rdt,
#                            n_estimators=1000
#                           )
#
# ./${PYTHON_SCRIPT_NAME} $TRAIN_SET_FILE_PATH $VALID_SET_FILE_PATH $TEST_SET_FILE_PATH 
#                         $ADV_BOOSTING_MODEL 1000 -l sse -o $MODELS_DIR -n 100 -d 8 -i 32
#                         -b 60 -a ../out/census_attacks
#
#######################################################################################
#
# 5. TRAINING UNATTACKED RANDOM FOREST
#   rdt = RobustDecisionTree(0, 
#                            attacker=EMPTY_ATTACKER,
#                            split_optimizer=OPTIMIZER, 
#                            max_depth=MAX_DEPTH, 
#                            min_instances_per_node=N_INSTANCES_PER_NODE, 
#                            max_samples=1.0,
#                            max_features=0.25,
#                            feature_blacklist=set([]),
#                           )
#   rf = RobustForest(0,
#                     base_estimator=rdt,
#                     n_estimators=1000
#                    )
#
# ./${PYTHON_SCRIPT_NAME} $TRAIN_SET_FILE_PATH $VALID_SET_FILE_PATH $TEST_SET_FILE_PATH 
#                         $STD_MODEL 1000 -l sse -o $MODELS_DIR -n 100 -d 8 -i 32
#                         -b 0 -a ../out/census_attacks
#
#######################################################################################
#
# 6. TRAINING ATTACKED RANDOM FOREST
#   rdt = RobustDecisionTree(0, 
#                            attacker=ATTACKER,
#                            split_optimizer=OPTIMIZER, 
#                            max_depth=MAX_DEPTH, 
#                            min_instances_per_node=N_INSTANCES_PER_NODE, 
#                            max_samples=1.0,
#                            max_features=0.25,
#                            feature_blacklist=set([]),
#                           )
#   rf = RobustForest(0,
#                     base_estimator=rdt,
#                     n_estimators=1000
#                    )
#
# ./${PYTHON_SCRIPT_NAME} $TRAIN_SET_FILE_PATH $VALID_SET_FILE_PATH $TEST_SET_FILE_PATH 
#                         $ROBUST_MODEL 1000 -l sse -o $MODELS_DIR -n 100 -d 8 -i 32
#                         -b 60 -a ../out/census_attacks
#
#######################################################################################
