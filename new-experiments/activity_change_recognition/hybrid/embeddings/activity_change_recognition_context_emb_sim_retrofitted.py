import json
import sys
import numpy as np
import pandas as pd
import argparse

from sklearn import preprocessing
from pylab import *
from math import sqrt

from utils.activity_change_preprocessing import *
from utils.activity_change_save_results import *
from utils.activity_change_evaluation import *

from embeddings_utils.create_embedding_matrix import *
from context_similarity.calculate_context_similarity import *

import multiprocessing
from gensim.models import Word2Vec

def main(argv):
    np.set_printoptions(threshold=sys.maxsize)
    np. set_printoptions(suppress=True)
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="../../kasteren_house_a",
                        nargs="?",
                        help="Dataset dir")
    parser.add_argument("--dataset_file",
                        type=str,
                        default="kasterenA_groundtruth.csv",
                        nargs="?",
                        help="Dataset file")
    parser.add_argument("--results_dir",
                        type=str,
                        default='results/kasteren_house_a',
                        nargs="?",
                        help="Dir for results")
    parser.add_argument("--results_folder",
                        type=str,
                        default='word2vec_context_retrofitted_location',
                        nargs="?",
                        help="Folder for results")
    parser.add_argument("--vector_file",
                        type=str,
                        default='0_execution_location_retrofitted.vector',
                        nargs="?",
                        help="Vector file with retrofitted action vectors")
    parser.add_argument("--train_or_test",
                        type=str,
                        default='train',
                        nargs="?",
                        help="Specify train or test data")
    parser.add_argument("--embedding_size",
                        type=int,
                        default=50,
                        nargs="?",
                        help="Embedding size for word2vec algorithm")
    parser.add_argument("--window_size",
                        type=int,
                        default=1,
                        nargs="?",
                        help="Window size for word2vec algorithm")
    parser.add_argument("--context_window_size",
                        type=int,
                        default=2,
                        nargs="?",
                        help="Context window size for CPD algorithm")
    parser.add_argument("--iterations",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Iterations for word2vec algorithm")
    parser.add_argument("--exe",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Number of executions")
    args = parser.parse_args()
    print('Loading dataset...')
    sys.stdout.flush()
    # dataset of actions and activities
    DATASET = args.dataset_dir + "/" + args.dataset_file.replace('.csv', '') + "_" + args.train_or_test + ".csv"
    df_dataset = pd.read_csv(DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]
    # list of unique actions in the dataset
    UNIQUE_ACTIONS = args.dataset_dir + "/" + 'unique_actions.json'
    # context information for the actions in the dataset
    CONTEXT_OF_ACTIONS = args.dataset_dir + "/" + 'context_model.json'
    # total actions and its names
    unique_actions = json.load(open(UNIQUE_ACTIONS, 'r'))
    total_actions = len(unique_actions)
    # context of actions
    context_of_actions = json.load(open(CONTEXT_OF_ACTIONS, 'r'))
    action_location = {}
    for key, values in context_of_actions['objects'].items():
        action_location[key] = values['location']
    # check action:location dict struct
    print(action_location)
    # check dataset struct
    print("Dataset")
    print(df_dataset)
    # prepare dataset
    X, timestamps, days, hours, seconds_past_midnight, y, tokenizer_action = prepare_x_y_activity_change(df_dataset)
    # transform action:location dict struct to action_index:location struct
    action_index = tokenizer_action.word_index
    action_index_location = {}
    for key, value in action_index.items():
        action_index_location[value] = action_location[key]
    # check action_index:location struct
    print(action_index_location)
    # check prepared dataset struct
    print("Actions")
    print(X)
    print("Activity change")
    print(y)
    # change point detection
    window_size = args.window_size
    context_window_size = args.context_window_size
    iterations = args.iterations
    exe = args.exe
    embedding_size = args.embedding_size
    vector_file = args.vector_file
    RESULTS_DIR = "/" + args.results_dir + "/" + args.results_folder + "/context_window_" + str(context_window_size) + "_window_" + str(window_size) + "_iterations_" + str(iterations) + "_embedding_size_" + str(embedding_size) + "/" + args.train_or_test + "/"
    # create dirs for saving results
    create_dirs(RESULTS_DIR, word2vec=True)
    print('Created dirs at: ' + RESULTS_DIR)
    # check actions input shape
    print("Input action shape: " + str(X.shape))
    # repeat exe iterations
    results_1 = np.zeros((4,10,30))
    results_5 = np.zeros((4,10,30))
    results_10 = np.zeros((4,10,30))
    detection_delays = np.zeros((10,30))
    models = []
    for e in range(0, exe):
        # create embedding matrix from word2vec retrofitted vector file
        embedding_action_matrix, unknown_actions = create_action_embedding_matrix_from_file(tokenizer_action, vector_file, embedding_size)
        # calculate context similarities using word2vec embeddings
        similarities = []
        similarities.append(0)
        for i in range(1, len(X)):
            context_similarity = calculate_context_similarity(X, embedding_action_matrix, i, context_window_size)
            similarities.append(context_similarity)
        # prepare change detection with offset using different max_context_distance values
        counter_max_context_distance = 0
        for max_context_distance in [x * 0.1 for x in range(0, 10)]:
            cf_matrix_1 = get_conf_matrix_with_offset_strategy(similarities, y, timestamps, max_context_distance, 1)
            cf_matrix_5 = get_conf_matrix_with_offset_strategy(similarities, y, timestamps, max_context_distance, 5)
            cf_matrix_10 = get_conf_matrix_with_offset_strategy(similarities, y, timestamps, max_context_distance, 10)
            # TPR, TNR, FPR, G-MEAN for exact change point detection
            TN, FP, FN, TP = cf_matrix_1.ravel()
            TPR = TP/(TP+FN)
            TNR = TN/(TN+FP)
            FPR = FP/(FP+TN)
            G_MEAN = sqrt(TPR * TNR)
            results_1[0][counter_max_context_distance][e] = TPR
            results_1[1][counter_max_context_distance][e] = TNR
            results_1[2][counter_max_context_distance][e] = FPR
            results_1[3][counter_max_context_distance][e] = G_MEAN
            # TPR, TNR, FPR, G-MEAN for 5 second offset change point detection
            TN, FP, FN, TP = cf_matrix_5.ravel()
            TPR = TP/(TP+FN)
            TNR = TN/(TN+FP)
            FPR = FP/(FP+TN)
            G_MEAN = sqrt(TPR * TNR)
            results_5[0][counter_max_context_distance][e] = TPR
            results_5[1][counter_max_context_distance][e] = TNR
            results_5[2][counter_max_context_distance][e] = FPR
            results_5[3][counter_max_context_distance][e] = G_MEAN
            # TPR, TNR, FPR, G-MEAN for 10 second offset change point detection
            TN, FP, FN, TP = cf_matrix_10.ravel()
            TPR = TP/(TP+FN)
            TNR = TN/(TN+FP)
            FPR = FP/(FP+TN)
            G_MEAN = sqrt(TPR * TNR)
            results_10[0][counter_max_context_distance][e] = TPR
            results_10[1][counter_max_context_distance][e] = TNR
            results_10[2][counter_max_context_distance][e] = FPR
            results_10[3][counter_max_context_distance][e] = G_MEAN
            # detection delay
            detection_delay = get_detection_delay(similarities, y, timestamps, max_context_distance)
            detection_delays[counter_max_context_distance][e] = detection_delay
            counter_max_context_distance += 1
    # save population of results to file
    for max_context_distance_num in range(0, 10):
        save_pop_results_to_file(RESULTS_DIR, results_1, 1, max_context_distance_num)
        save_pop_results_to_file(RESULTS_DIR, results_5, 5, max_context_distance_num)
        save_pop_results_to_file(RESULTS_DIR, results_10, 10, max_context_distance_num)
        np.savetxt(RESULTS_DIR + 'detection_delays/' + str(max_context_distance_num) + '_detection_delay' + '.csv', detection_delays[max_context_distance_num])
    # save trained models
    model_num = 0
    for model in models:
        model.save(RESULTS_DIR + 'word2vec_models/' + str(model_num) + '_execution.model')
        model_num += 1
    # mark experiment end
    print('... Experiment finished ...')
    print('Results saved to: ' + RESULTS_DIR)
    print('... ... ... ... ... ... ...')

if __name__ == "__main__":
    main(sys.argv)