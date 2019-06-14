#!/bin/bash

# This is the top-level (i.e., outermost) script, which is used to launch a set of training runs
# using several different configurations of hyperparameters.

# Remember to cite GNU Parallel as follows:
# @book{tange_ole_2018_1146014,
#       author       = {Tange, Ole},
#       title        = {GNU Parallel 2018},
#       publisher    = {Ole Tange},
#       month        = Mar,
#       year         = 2018,
#       ISBN         = {9781387509881},
#       doi          = {10.5281/zenodo.1146014},
#       url          = {https://doi.org/10.5281/zenodo.1146014}
# }

if [ $# -lt 1 ]
then
    echo "Wrong number of input arguments supplied! Please, invoke the script as follows:"
    echo ""
    echo "> ./run_train.sh <DATASET_NAME>"
    echo ""
    echo "where:"
    echo "- <DATASET_NAME> is the name of the dataset used for this training run (e.g., census)"
    echo ""
    exit 1
fi


PARALLEL=1

DATASET_NAME=$1
BASH_SCRIPT_NAME="train_robust_forest"
BASH_SCRIPT="${BASH_SCRIPT_NAME}.sh"

ALGO=(par-robust)
N_ESTIMATORS=(100)
DEPTHS=(8)
BUDGETS=(45)


echo "*********** Training Robust Random Forest on ${DATASET_NAME} dataset ***********"
echo ""

echo "parallel --eta --bar -j ${PARALLEL} --joblog /tmp/${BASH_SCRIPT_NAME}_${DATASET_NAME}.log ./${BASH_SCRIPT} {1} {2} {3} {4} {5} ::: "${DATASET_NAME}" ::: "${ALGO[@]}" ::: "${N_ESTIMATORS[@]}" ::: "${DEPTHS[@]}" ::: "${BUDGETS[@]}""
echo ""
parallel --eta --bar -j ${PARALLEL} --joblog /tmp/${BASH_SCRIPT_NAME}_${DATASET_NAME}.log ./${BASH_SCRIPT} {1} {2} {3} {4} {5} ::: "${DATASET_NAME}" ::: "${ALGO[@]}" ::: "${N_ESTIMATORS[@]}" ::: "${DEPTHS[@]}" ::: "${BUDGETS[@]}"
echo ""
echo "**************************************************************"
