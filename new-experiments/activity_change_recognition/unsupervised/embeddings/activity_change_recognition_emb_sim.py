import json
import sys
import os
import numpy as np
import pandas as pd
import ruptures as rpt
import itertools
import operator
import csv
import argparse

from gensim.models import Word2Vec
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from scipy import spatial
from scipy.stats import entropy, norm
from datetime import datetime as dt

from pylab import *
from scipy import linalg
from scipy.stats import norm

from math import sqrt

import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Kasteren dataset DIR
DIR = '../../kasteren_house_a/'
# Kasteren dataset file
DATASET_CSV = DIR + 'base_kasteren_reduced'
# List of unique actions in the dataset
UNIQUE_ACTIONS = DIR + 'unique_actions.json'
# Context information for the actions in the dataset
CONTEXT_OF_ACTIONS = DIR + 'context_model.json'

def create_dirs(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir + "1s", exist_ok=True)
    os.makedirs(results_dir + "5s", exist_ok=True)
    os.makedirs(results_dir + "10s", exist_ok=True)
    os.makedirs(results_dir + "detection_delays", exist_ok=True)
    os.makedirs(results_dir + "word2vec_models", exist_ok=True)

def get_unixtime(dt64):
    return dt64.astype('datetime64[s]').astype('int')

def convert_epoch_time_to_hour_of_day(epoch_time_in_seconds):
    d = dt.fromtimestamp(epoch_time_in_seconds)
    return d.strftime('%H')

def convert_epoch_time_to_day_of_the_week(epoch_time_in_seconds):
    d = dt.fromtimestamp(epoch_time_in_seconds)
    return d.strftime('%A')

def get_seconds_past_midnight_from_epoch(epoch_time_in_seconds):
    date = dt.fromtimestamp(epoch_time_in_seconds)
    seconds_past_midnight = (date - date.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    return seconds_past_midnight

def create_action_embedding_matrix(tokenizer, model, embedding_size):  
    action_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(action_index) + 1, embedding_size))
    unknown_actions = {}    
    for action, i in list(action_index.items()):
        try:            
            embedding_vector = model[action]
            embedding_matrix[i] = embedding_vector            
        except:
            if action in unknown_actions:
                unknown_actions[action] += 1
            else:
                unknown_actions[action] = 1
    
    return embedding_matrix, unknown_actions

def prepare_x_y_activity_change(df):
    actions = df['action'].values
    dates = list(df.index.values)
    timestamps = list(map(get_unixtime, dates))
    hours = list(map(convert_epoch_time_to_hour_of_day, timestamps))
    days = list(map(convert_epoch_time_to_day_of_the_week, timestamps))
    seconds_past_midnight = list(map(get_seconds_past_midnight_from_epoch, timestamps))
    activities = df['activity'].values

    tokenizer_action = Tokenizer(lower=False)
    tokenizer_action.fit_on_texts(actions.tolist())
    action_index = tokenizer_action.word_index
    
    actions_by_index = []
    for action in actions:
        actions_by_index.append(action_index[action])
    
    X = []
    y = []
    last_activity = None
    for i in range(0, len(actions)):
        X.append(actions_by_index[i])
        if (i == 0):
            y.append(0)
        elif last_activity == activities[i]:
            y.append(0)
        else:
            y.append(1)
        last_activity = activities[i]
    
    X = np.array(X)
    timestamps = np.array(timestamps)
    days = np.array(days)
    hours = np.array(hours)
    seconds_past_midnight = np.array(seconds_past_midnight)
    y = np.array(y)
    
    return X, timestamps, days, hours, seconds_past_midnight, y, tokenizer_action

def check_detected_change_point_with_offset(timestamps, y, position, offset):
    timestamp_ref = timestamps[position]

    # right
    for i in range(position, len(y)):
        difference = timestamps[i] - timestamp_ref
        if offset < difference:
            break
        else:
            if y[i] == 1:
                return True
    # left
    i = position - 1
    while i >= 0:
        difference = timestamp_ref - timestamps[i]
        if offset < difference:
            break
        else:
            if y[i] == 1:
                return True
        i -= 1

    return False

def get_conf_matrix_with_offset_strategy(similarities, y, timestamps, min_cosine_distance, offset):
    cf_matrix = np.zeros((2,2))

    for i in range(0, len(similarities)):
        if similarities[i] < min_cosine_distance:
            correctly_detected = check_detected_change_point_with_offset(timestamps, y, i, offset)
            if correctly_detected:
                cf_matrix[1][1] += 1
            else:
                cf_matrix[0][1] += 1
        else:
            if y[i] == 0:
                cf_matrix[0][0] += 1
            else:
                cf_matrix[1][0] += 1
    
    return cf_matrix

def search_nearest_change_point(y, timestamps, position):
    timestamp_ref = timestamps[position]

    # right
    right_nearest_cp_difference = sys.maxsize
    for i in range(position, len(y)):
        if y[i] == 1:
            right_nearest_cp_difference = timestamps[i] - timestamp_ref
            break
    
    # left
    left_nearest_cp_difference = sys.maxsize
    i = position - 1
    while i >= 0:
        if y[i] == 1:
            left_nearest_cp_difference = timestamp_ref - timestamps[i]
            break
        i -= 1

    return min(right_nearest_cp_difference, left_nearest_cp_difference)

def get_detection_delay(similarities, y, timestamps, min_cosine_distance):
    detection_delay = 0

    counter = 0
    for i in range(0, len(similarities)):
        if similarities[i] > min_cosine_distance:
            counter += 1
            detection_delay += search_nearest_change_point(y, timestamps, i)
    
    if counter == 0: 
        return 0
    else:
        return detection_delay / counter

def save_pop_results_to_file(results_dir, results, seconds_for_detection, min_cosine_distance_num):
    np.savetxt(results_dir + str(seconds_for_detection) + 's/results_' + str(seconds_for_detection) + '_' + str(min_cosine_distance_num) + '_TPR' + '.csv', results[0][min_cosine_distance_num])
    np.savetxt(results_dir + str(seconds_for_detection) + 's/results_' + str(seconds_for_detection) + '_' + str(min_cosine_distance_num) + '_TNR' + '.csv', results[1][min_cosine_distance_num])
    np.savetxt(results_dir + str(seconds_for_detection) + 's/results_' + str(seconds_for_detection) + '_' + str(min_cosine_distance_num) + '_FPR' + '.csv', results[2][min_cosine_distance_num])
    np.savetxt(results_dir + str(seconds_for_detection) + 's/results_' + str(seconds_for_detection) + '_' + str(min_cosine_distance_num) + '_G-MEAN' + '.csv', results[3][min_cosine_distance_num])

def main(argv):
    np.set_printoptions(threshold=sys.maxsize)
    np. set_printoptions(suppress=True)
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",
                        type=str,
                        default='results',
                        nargs="?",
                        help="Dir for results")
    parser.add_argument("--results_folder",
                        type=str,
                        default='word2vec',
                        nargs="?",
                        help="Folder for results")
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
    parser.add_argument("--iterations",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Iterations for word2vec algorithm")
    parser.add_argument("--exe",
                        type=int,
                        default=30,
                        nargs="?",
                        help="Number of executions")
    args = parser.parse_args()
    print('Loading dataset...')
    sys.stdout.flush()
    # dataset of actions and activities
    DATASET = DATASET_CSV + "_" + args.train_or_test + ".csv"
    df_dataset = pd.read_csv(DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]
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
    iterations = args.iterations
    exe = args.exe
    embedding_size = args.embedding_size
    RESULTS_DIR = "/" + args.results_dir + "/" + args.results_folder + "/window_" + str(window_size) + "_iterations_" + str(iterations) + "_embedding_size_" + str(embedding_size) + "/" + args.train_or_test + "/"
    # create dirs for saving results
    create_dirs(RESULTS_DIR)
    # check actions input shape
    print("Input action shape: " + str(X.shape))
    # repeat exe iterations
    results_1 = np.zeros((4,10,30))
    results_5 = np.zeros((4,10,30))
    results_10 = np.zeros((4,10,30))
    detection_delays = np.zeros((10,30))
    models = []
    for e in range(0, exe):
        # if train set, then train word2vec model and save it
        if args.train_or_test == 'train':
            model = Word2Vec([df_dataset['action'].values.tolist()],
                size=embedding_size, window=window_size, min_count=0, iter=iterations,
                workers=multiprocessing.cpu_count())
            models.append(model)
        # if test set, load word2vec model
        elif args.train_or_test == 'test':
            model = Word2Vec.load(RESULTS_DIR + 'word2vec_models/' + str(exe) + '_execution.model') #TODO check this
        # create embedding matrix
        embedding_action_matrix, unknown_actions = create_action_embedding_matrix(tokenizer_action, model, embedding_size)
        # calculate similarities using word2vec embeddings
        similarities = []
        for i in range(0, len(X)-1):
            similarity = 1 - spatial.distance.cosine(embedding_action_matrix[X[i]], embedding_action_matrix[X[i+1]])
            similarities.append(similarity)
        # prepare change detection with offset using different min_cosine_distance values
        counter_min_cosine_distance = 0
        for min_cosine_distance in [x * 0.1 for x in range(0, 10)]:
            cf_matrix_1 = get_conf_matrix_with_offset_strategy(similarities, y, timestamps, min_cosine_distance, 1)
            cf_matrix_5 = get_conf_matrix_with_offset_strategy(similarities, y, timestamps, min_cosine_distance, 5)
            cf_matrix_10 = get_conf_matrix_with_offset_strategy(similarities, y, timestamps, min_cosine_distance, 10)
            # TPR, TNR, FPR, G-MEAN for exact change point detection
            TN, FP, FN, TP = cf_matrix_1.ravel()
            TPR = TP/(TP+FN)
            TNR = TN/(TN+FP)
            FPR = FP/(FP+TN)
            G_MEAN = sqrt(TPR * TNR)
            results_1[0][counter_min_cosine_distance][e] = TPR
            results_1[1][counter_min_cosine_distance][e] = TNR
            results_1[2][counter_min_cosine_distance][e] = FPR
            results_1[3][counter_min_cosine_distance][e] = G_MEAN
            # TPR, TNR, FPR, G-MEAN for 5 second offset change point detection
            TN, FP, FN, TP = cf_matrix_5.ravel()
            TPR = TP/(TP+FN)
            TNR = TN/(TN+FP)
            FPR = FP/(FP+TN)
            G_MEAN = sqrt(TPR * TNR)
            results_5[0][counter_min_cosine_distance][e] = TPR
            results_5[1][counter_min_cosine_distance][e] = TNR
            results_5[2][counter_min_cosine_distance][e] = FPR
            results_5[3][counter_min_cosine_distance][e] = G_MEAN
            # TPR, TNR, FPR, G-MEAN for 10 second offset change point detection
            TN, FP, FN, TP = cf_matrix_10.ravel()
            TPR = TP/(TP+FN)
            TNR = TN/(TN+FP)
            FPR = FP/(FP+TN)
            G_MEAN = sqrt(TPR * TNR)
            results_10[0][counter_min_cosine_distance][e] = TPR
            results_10[1][counter_min_cosine_distance][e] = TNR
            results_10[2][counter_min_cosine_distance][e] = FPR
            results_10[3][counter_min_cosine_distance][e] = G_MEAN
            # detection delay
            detection_delay = get_detection_delay(similarities, y, timestamps, min_cosine_distance)
            detection_delays[counter_min_cosine_distance][e] = detection_delay
            counter_min_cosine_distance += 1
    # save population of results to file
    for min_cosine_distance_num in range(0, 10):
        save_pop_results_to_file(RESULTS_DIR, results_1, 1, min_cosine_distance_num)
        save_pop_results_to_file(RESULTS_DIR, results_5, 5, min_cosine_distance_num)
        save_pop_results_to_file(RESULTS_DIR, results_10, 10, min_cosine_distance_num)
        np.savetxt(RESULTS_DIR + 'detection_delays/' + str(min_cosine_distance_num) + '_detection_delay' + '.csv', detection_delays[min_cosine_distance_num])
    # save trained models
    model_num = 0
    for model in models:
        model.save(RESULTS_DIR + 'word2vec_models/' + str(model_num) + '_execution.model')
        model_num += 1

if __name__ == "__main__":
    main(sys.argv)