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


#mkdir -p out/models

PARALLEL=16
#PARALLEL=4

TRAIN=./train_robust_forest.sh

# 1 tree already done
N_ESTIMATORS=(10)
BUDGETS=(30)
DEPTHS=(8)
ALGO=(robust)

# standard 1 0 8 [attack-unaware standard decision tree]
# robust 1 60 8 [attack-aware robust decision tree]
# standard 1000 0 8 [attack-unaware standard random forest]
# reduced 1000 0 8 [attack-unaware reduced random forest]
# adv-boosting 1000 60 8 [attack-aware adversarial boosting]
# robust 1000 60 8 [attack-aware robust random forest]



for d in ${DEPTHS[@]}
do  
    for a in ${ALGO[@]}
    do  
        if [ "$a" = "standard" ]; then
            for e in ${N_ESTIMATORS[@]}
            do
                CMD="";
                CMD="${TRAIN} ${a} ${e} ${d} 0";
                    
                echo executing $CMD;
                sem -j ${PARALLEL} ${CMD};
            done
                
        elif [[ "$a" = "reduced" ]]; then
            for e in ${N_ESTIMATORS[@]}
            do
                if [ "$e" -gt 1 ]
                then
                    CMD="";
                    CMD="${TRAIN} ${a} ${e} ${d} 0";
                    
                    echo executing $CMD;
                    sem -j ${PARALLEL} ${CMD};
                fi
            done

        elif [[ "$a" = "adv-boosting" ]]; then
            for e in ${N_ESTIMATORS[@]}
            do
                if [ "$e" -gt 1 ]
                then
                    CMD="";
                    for b in ${BUDGETS[@]}
                    do
                        CMD="${TRAIN} ${a} ${e} ${d} ${b}";
                    
                        echo executing $CMD;
                        sem -j ${PARALLEL} ${CMD};
                    done
                fi
            done

        elif [[ "$a" = "robust" ]]; then
            for e in ${N_ESTIMATORS[@]}
            do
                CMD="";
                for b in ${BUDGETS[@]}
                do
                    CMD="${TRAIN} ${a} ${e} ${d} ${b}";
                    
                    echo executing $CMD;
                    sem -j ${PARALLEL} ${CMD};
                done
            done

        fi
    done
done
sem --wait
