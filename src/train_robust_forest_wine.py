#!/usr/bin/env python3
# encoding: utf-8

"""
train_robust_forest.py

Created by Gabriele Tolomei on 2019-01-23.
"""

import sys
import os
import argparse
import logging
import numpy as np
import pandas as pd
import robust_forest as rf

# logging.basicConfig(
#     format='%(asctime)s : %(levelname)s : %(message)s', level=logger.info)


def configure_logging():
    """
    Logging setup
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    LOGGING_FORMAT = '%(asctime)-15s *** %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s()] *** %(message)s'
    formatter = logging.Formatter(LOGGING_FORMAT)

    # log to stdout console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # log to file
    file_handler = logging.FileHandler(
        filename="./train_robust_forest_wine.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_options(cmd_args=None):
    """
    Parse command line arguments
    """
    cmd_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmd_parser.add_argument(
        'training_set',
        help="""Path to the file containing the training set.""",
        type=str)
    cmd_parser.add_argument(
        'valid_set',
        help="""Path to the file containing the validation set.""",
        type=str)
    cmd_parser.add_argument(
        'test_set',
        help="""Path to the file containing the test set.""",
        type=str)
    cmd_parser.add_argument(
        'model_type',
        default='standard',
        const='standard',
        nargs='?',
        choices=['standard', 'reduced', 'adv-boosting', 'robust'],
        help="""List of possible models to train (default = standard).""")
    cmd_parser.add_argument(
        'n_estimators',
        help="""Number of base estimators used for training the ensemble of trees (default = 1000).""",
        type=is_strictly_positive,
        default=1000)
    cmd_parser.add_argument(
        '-l',
        '--loss-function',
        default='sse',
        const='sse',
        nargs='?',
        choices=['sse', 'mse', 'mae', 'gini_impurity', 'entropy'],
        help="""List of possible loss function used at training time (default = sse).""")
    cmd_parser.add_argument(
        '-o',
        '--output_dirname',
        help="""Path to the output directory containing results.""",
        type=str,
        default='./')
    cmd_parser.add_argument(
        '-n',
        '--n_instances',
        help="""Number of instances used for training (default = 1000).""",
        type=is_strictly_positive,
        default=1000)
    cmd_parser.add_argument(
        '-bs',
        '--bootstrap_samples',
        help="""Percentage of instances randomly sampled (without replacement) for each base estimator.""",
        type=is_valid_percentage,
        default=100)
    cmd_parser.add_argument(
        '-bf',
        '--bootstrap_features',
        help="""Percentage of features randomly sampled (without replacement) at each node split.""",
        type=is_valid_percentage,
        default=100)
    cmd_parser.add_argument(
        '-xf',
        '--exclude_features',
        nargs='*',
        help="""List of feature names (i.e., column names) to be excluded during training.""")
    cmd_parser.add_argument(
        '-d',
        '--max_depth',
        help="""Maximum depth of the tree (default = 4).""",
        type=is_non_negative,
        default=4)
    cmd_parser.add_argument(
        '-i',
        '--instances_per_node',
        help="""Minimum number of instances per node to check for further split (default = 20).""",
        type=is_strictly_positive,
        default=20)
    cmd_parser.add_argument(
        '-b',
        '--attacker_budget',
        help="""Maximum allowed budget for the attacker (default = 60).""",
        type=is_non_negative,
        default=60)
    cmd_parser.add_argument(
        '-a',
        '--attacks_filename',
        help="""Path to the file with attacks.""",
        type=str,
        default='./data/attacks')

    args = cmd_parser.parse_args(cmd_args)

    options = {}
    options['training_set'] = args.training_set
    options['valid_set'] = args.valid_set
    options['test_set'] = args.test_set
    options['model_type'] = args.model_type
    options['n_estimators'] = args.n_estimators
    options['loss_function'] = args.loss_function
    options['output_dirname'] = args.output_dirname
    options['n_instances'] = args.n_instances
    options['max_depth'] = args.max_depth
    options['bootstrap_samples'] = args.bootstrap_samples
    options['bootstrap_features'] = args.bootstrap_features
    options['exclude_features'] = args.exclude_features
    options['instances_per_node'] = args.instances_per_node
    options['attacker_budget'] = args.attacker_budget
    options['attacks_filename'] = args.attacks_filename

    return options

######################## Checking and validating input parameters ############


def is_valid_percentage(value):
    """
    This function is responsible for checking the validity of input arguments.

    Args:
            value (str): value passed as input argument to this script

    Return:
            an int if value is such that 0 <= value <= 100, an argparse.ArgumentTypeError otherwise
    """
    ivalue = int(value)
    if ivalue < 0 or ivalue > 100:
        raise argparse.ArgumentTypeError(
            "{} is an invalid percentage value for the input argument which must be any x, such that 0 <= x <= 100".format(ivalue))
    return ivalue


def is_non_negative(value):
    """
    This function is responsible for checking the validity of input arguments.

    Args:
            value (str): value passed as input argument to this script

    Return:
            an int if value is such that value >= 0, an argparse.ArgumentTypeError otherwise
    """
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            "{} is an invalid value for the input argument which must be any x, such that x >= 0".format(ivalue))
    return ivalue


def is_strictly_positive(value):
    """
    This function is responsible for checking the validity of input arguments.

    Args:
            value (str): value passed as input argument to this script

    Return:
            an int if value is such that value > 0, an argparse.ArgumentTypeError otherwise
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "{} is an invalid value for the input argument which must be any x, such that x > 0".format(ivalue))
    return ivalue

###########################################################################


def create_attack_rules(dataset, colnames):
    # pre_conditions = {feature_id: (value_left, value_right)}
    # post_condition = {feature_id: new_value}
    # cost

    attack_rules = []

    ########################### ALCOHOL ###########################

    attack_rules.append(rf.AttackerRule(
        {colnames.index("alcohol"): ((0, 10))},
        {colnames.index("alcohol"): 0.75},
        cost=10,
        is_numerical=True
    ))
    ########################### RESIDUAL SUGAR ###########################

    attack_rules.append(rf.AttackerRule(
        {colnames.index("residual_sugar"): ((8, np.inf))},
        {colnames.index("residual_sugar"): -1.2},
        cost=10,
        is_numerical=True
    ))
    ########################### VOLATILE ACIDITY ###########################

    attack_rules.append(rf.AttackerRule(
        {colnames.index("volatile_acidity"): ((0.6, np.inf))},
        {colnames.index("volatile_acidity"): -0.3},
        cost=10,
        is_numerical=True
    ))

    return attack_rules

##########################################################################


def build_output_model_filename(dataset_name, model_type, n_estimators, max_depth, instances_per_node, budget, ext="model"):
    # attacked = ""
    # if budget == 0:
    #     attacked = "unattacked"
    # else:
    #     attacked = "attacked-{}".format(budget)

    return "{}_{}_B{}_T{}_D{}_I{}.{}".format(model_type, dataset_name, budget, n_estimators, max_depth, instances_per_node, ext)

##########################################################################


def get_attacked_dataset(model, attacker, X, y):

    logger = logging.getLogger(__name__)
    # array of arrays of attacks (sorted by increasing cost)
    logger.info("Generating attacks...")
    attacks = [sorted(attacker.attack(X[i], 0), key=lambda x: x[1])
               for i in range(X.shape[0])]
    # array of arrays of model predictions
    logger.info("Computing predictions...")
    preds = []
    for x_attack in attacks:
        preds.append([model.predict(x_atk[0].reshape(1, X.shape[1]))[0]
                      for x_atk in x_attack])
    i = 0
    X_att = []
    y_att = []
    logger.info("Building the attacked dataset...")
    while i < len(preds):
        cur_pred = preds[i][0]
        j = 0
        while j < len(attacks[i]):
            if(preds[i][j] != cur_pred):
                X_att.append(attacks[i][j][0])
                y_att.append(y[i])
                break
            j += 1
        i += 1

    return np.array(X_att).reshape(len(X_att), X.shape[1]), np.array(y_att)

##########################################################################


def loading_dataset(dataset_filename, sep=","):

    return pd.read_csv(dataset_filename, sep=sep)

##########################################################################


############################ Main ########################################


def main(options):

    logger = configure_logging()

    logger.info("==> Loading training set from " + options['training_set'])
    train = loading_dataset(options['training_set'])

    logger.info(
        "- Shape of the training set: number of instances = {}; number of features = {} ({} is the label)".format(
            train.shape[0],
            train.shape[1] - 1,
            train.shape[1]))

    logger.info("==> Loading test set from " + options['test_set'])
    test = loading_dataset(options['test_set'])

    logger.info(
        "- Shape of the test set: number of instances = {}; number of features = {} ({} is the label)".format(
            test.shape[0],
            test.shape[1] - 1,
            test.shape[1]))

    logger.info("==> Extract column names...")
    # column names
    colnames = train.columns.tolist()

    logger.info("==> Encoding attack rules...")
    attack_rules = create_attack_rules(train, colnames)
    logger.info("==> Create the corresponding attacker...")
    attacker = rf.Attacker(
        attack_rules, options['attacker_budget'])

    logger.info("==> Extract feature matrix from {} instances of training set".format(
        options['n_instances']))
    X_train = train.iloc[:, :-1].values[:options['n_instances']]
    logger.info("==> Extract label vector from {} instances of training set".format(
        options['n_instances']))
    y_train = train.iloc[:, -1].replace(-1, 0).values[:options['n_instances']]

    attacker.attack_dataset(
        X_train, attacks_filename='{}_B{}.atks'.format(options['attacks_filename'], str(options['attacker_budget'])))

    feature_blacklist = {}
    if options['exclude_features']:
        logger.info("==> Excluding the following features from training: [{}]".format(
            ", ".join([f for f in options['exclude_features']])))
        feature_blacklist_names = options['exclude_features']
        feature_blacklist_ids = [colnames.index(
            fb) for fb in feature_blacklist_names]
        feature_blacklist = dict(
            zip(feature_blacklist_ids, feature_blacklist_names))

    dataset_name = options['attacks_filename'].split('/')[-1].split('_')[0]
    output_model_filename = build_output_model_filename(
        dataset_name, options['model_type'], options['n_estimators'], options['max_depth'], options['instances_per_node'], options['attacker_budget'])
    partial_output_model_filename = output_model_filename.split('.')[0]

    logger.info(
        "==> Create the split optimizer which will be used for this training...")
    optimizer = rf.SplitOptimizer(
        split_function=rf.SplitOptimizer._SplitOptimizer__sse, split_function_name=options['loss_function'])

    rdt = rf.RobustDecisionTree(0,
                                attacker=attacker,
                                split_optimizer=optimizer,
                                max_depth=options['max_depth'],
                                min_instances_per_node=options['instances_per_node'],
                                max_samples=options['bootstrap_samples'] / 100.0,
                                max_features=options['bootstrap_features'] / 100.0,
                                feature_blacklist=feature_blacklist
                                )

    logger.info("==> Training \"{}\" random forest...".format(
        options['model_type']))
    # create the robust forest
    rrf = rf.RobustForest(
        0, base_estimator=rdt, n_estimators=options['n_estimators'])
    rrf.fit(X_train, y=y_train,
            dump_filename=options['output_dirname'] + '/' + partial_output_model_filename, dump_n_trees=10)

    logger.info("==> Eventually, serialize the \"{}\" random forest just trained to {}".format(
        options['model_type'], options['output_dirname'] + '/' + output_model_filename))
    rrf.save(options['output_dirname'] + '/' + output_model_filename)


if __name__ == "__main__":
    sys.exit(main(get_options()))
