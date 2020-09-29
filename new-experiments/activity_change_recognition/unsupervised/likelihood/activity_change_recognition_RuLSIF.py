import json
import sys
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

from densratio import densratio

# Kasteren dataset DIR
DIR = '../../kasteren_house_a/'
# Kasteren dataset file
DATASET_CSV = DIR + 'base_kasteren_reduced'
# List of unique actions in the dataset
UNIQUE_ACTIONS = DIR + 'unique_actions.json'
# Context information for the actions in the dataset
CONTEXT_OF_ACTIONS = DIR + 'context_model.json'

##################################################################################################################
# Feature extraction approach of Real-Time Change Point Detection with Application to Smart Home Time Series Data
# https://ieeexplore.ieee.org/document/8395405
# START
##################################################################################################################

def get_unixtime(dt64):
    return dt64.astype('datetime64[s]').astype('int')

def most_common(L):
  SL = sorted((x, i) for i, x in enumerate(L))

  groups = itertools.groupby(SL, key=operator.itemgetter(0))

  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    return count, -min_index

  return max(groups, key=_auxfun)[0]

def calc_entropy(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)

def day_to_int(day):
    switcher = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    return switcher[day]

def location_to_int(location):
    switcher = {"Bathroom": 0, "Kitchen": 1, "Bedroom": 2, "Hall": 3}
    return switcher[location]

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

def extract_actions_from_window(actions, timestamps, position, window_size=30):
    actions_from_window = []
    timestamps_from_window = []

    for i in range(position-(window_size), position):
        if i >= 0:
            actions_from_window.append(actions[i])
            timestamps_from_window.append(timestamps[i])
    actions_from_window.append(actions[position])
    timestamps_from_window.append(timestamps[position])

    return actions_from_window, timestamps_from_window

def extract_features_from_window(actions, previous_actions_1, previous_actions_2, locations, timestamps):
    features_from_window = []
    
    most_recent_sensor = actions[len(actions)-1]
    first_sensor_in_window = actions[0]
    window_duration = timestamps[len(actions)-1] - timestamps[0]
    if previous_actions_1 is not None:
        most_frequent_sensor_1 = most_common(previous_actions_1)
    else:
        most_frequent_sensor_1 = 0 # non-existing sensor index
    if previous_actions_2 is not None:
        most_frequent_sensor_2 = most_common(previous_actions_2)
    else:
        most_frequent_sensor_2 = 0 # non-existing sensor index
    last_sensor_location = location_to_int(locations[actions[len(actions)-1]])
    # last_motion_sensor_location -> no aplica en kasteren !!!!!
    entropy_based_data_complexity = calc_entropy(actions)
    time_elapsed_since_last_sensor_event = timestamps[len(actions)-1] - timestamps[len(actions)-2]

    features_from_window.append(most_recent_sensor)
    features_from_window.append(first_sensor_in_window)
    features_from_window.append(window_duration)
    features_from_window.append(most_frequent_sensor_1)
    features_from_window.append(most_frequent_sensor_2)
    features_from_window.append(last_sensor_location)
    features_from_window.append(entropy_based_data_complexity)
    features_from_window.append(time_elapsed_since_last_sensor_event)

    return features_from_window

def extract_features_from_sensors(actions, unique_actions, all_actions, position, timestamps, all_timestamps):
    features_from_sensors = []
    
    # count of events for each sensor in window
    for action in unique_actions:
        counter = 0
        for action_fired in actions:
            if action == action_fired:
                counter += 1
        features_from_sensors.append(counter)
    
    # elapsed time for each sensor since last event
    found_actions = []
    counter = position
    for action in unique_actions:
        while(counter > 0):
            if action == all_actions[counter]:
                found_actions.append(action)
                features_from_sensors.append(timestamps[len(timestamps)-1] - all_timestamps[counter])
                break
            counter -= 1
        if action not in found_actions:
            features_from_sensors.append(all_timestamps[len(all_timestamps)-1] - all_timestamps[0]) # maximun time possible
        counter = position

    return features_from_sensors

##################################################################################################################
# Feature extraction approach of Real-Time Change Point Detection with Application to Smart Home Time Series Data
# https://ieeexplore.ieee.org/document/8395405
# END
##################################################################################################################

##################################################################################################################
# R_ULSIF based CPD algorithm translation from http://allmodelsarewrong.net/software.html
# Adapted for Kasteren dataset feature extraction based on Aminikhanghahi et al. paper
# START
##################################################################################################################  

def sliding_window_with_features(actions, unique_actions, locations, timestamps, days, hours, seconds_past_midnight, window_size, step):
    windows = None
    previous_actions_1 = None
    previous_actions_2 = None
    num_samples = len(actions)

    for i in range(0, num_samples):
        if (i % step == 0):
            offset = window_size * step
            if i + offset > num_samples:
                break
            window_actions = actions[i:(i+offset)]
            window_timestamps = timestamps[i:(i+offset)]
            # get feature vector from window
            feature_vector = []
            # time features
            feature_vector.append(int(hours[i]))
            feature_vector.append(day_to_int(days[i]))
            feature_vector.append(seconds_past_midnight[i])
            # window features
            window_features = extract_features_from_window(window_actions, previous_actions_1, previous_actions_2, locations, window_timestamps)
            feature_vector.extend(window_features)
            # sensor features
            sensor_features = extract_features_from_sensors(window_actions, unique_actions, actions, i, window_timestamps, timestamps)
            feature_vector.extend(sensor_features)
            # update previous actions
            previous_actions_2 = previous_actions_1
            previous_actions_1 = window_actions
            # add to windows struct
            feature_vector = np.array(feature_vector)
            if windows is None:
                windows = np.zeros((len(feature_vector), num_samples + 1 - window_size * step))
                print("Allocated window struct shape: " + str(windows.shape))
            windows[:,int(ceil(i/step))] = feature_vector.reshape(1,-1)
    
    windows = np.array(windows)

    return windows
            
def change_detection(actions, unique_actions, locations, timestamps, days, hours, seconds_past_midnight, n, k, alpha, fold):
    scores = []
    
    windows = sliding_window_with_features(actions, unique_actions, locations, timestamps,
        days, hours, seconds_past_midnight, k, 1)
    num_samples = windows.shape[1]
    print("Num window samples in change detection: " + str(num_samples))
    t = n

    while((t+n) <= num_samples):
        y = windows[:,(t-n):(n+t)]
        y_ref = y[:,0:n]
        y_test = y[:,n:]

        densratio_obj = densratio(y_test, y_ref, alpha=alpha)

        scores.append(densratio_obj.alpha_PE)

        t += 1
    
    print("Num of scores: " + str(len(scores)))

    return scores

##################################################################################################################
# R_ULSIF based CPD algorithm translation from MatLab code http://allmodelsarewrong.net/software.html
# Adapted for Kasteren dataset feature extraction based on Aminikhanghahi et al. paper
# END
################################################################################################################## 

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

def get_conf_matrix_with_offset_strategy(scores, y, timestamps, threshold, offset):
    cf_matrix = np.zeros((2,2))

    for i in range(0, len(scores)):
        if scores[i] > threshold:
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

def get_detection_delay(scores, y, timestamps, threshold):
    detection_delay = 0

    counter = 0
    for i in range(0, len(scores)):
        if scores[i] > threshold:
            counter += 1
            detection_delay += search_nearest_change_point(y, timestamps, i)
    
    return detection_delay / counter

def save_pop_results_to_file(results, seconds_for_detection, threshold_num):
    np.savetxt(RESULTS_DIR + str(seconds_for_detection) + 's/results_' + str(seconds_for_detection) + '_' + str(threshold_num) + '_TPR' + '.csv', results[0][threshold_num])
    np.savetxt(RESULTS_DIR + str(seconds_for_detection) + 's/results_' + str(seconds_for_detection) + '_' + str(threshold_num) + '_TNR' + '.csv', results[1][threshold_num])
    np.savetxt(RESULTS_DIR + str(seconds_for_detection) + 's/results_' + str(seconds_for_detection) + '_' + str(threshold_num) + '_FPR' + '.csv', results[2][threshold_num])
    np.savetxt(RESULTS_DIR + str(seconds_for_detection) + 's/results_' + str(seconds_for_detection) + '_' + str(threshold_num) + '_G-MEAN' + '.csv', results[3][threshold_num])

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
                        default='RuLSIF',
                        nargs="?",
                        help="Folder for results")
    parser.add_argument("--train_or_test",
                        type=str,
                        default='train',
                        nargs="?",
                        help="Specify train or test data")
    parser.add_argument("--n",
                        type=int,
                        default=2,
                        nargs="?",
                        help="Window size for n-real time algorithm")
    parser.add_argument("--k",
                        type=int,
                        default=30,
                        nargs="?",
                        help="Number of events to look when performing feature extraction")
    parser.add_argument("--alpha",
                        type=float,
                        default=0.01,
                        nargs="?",
                        help="RulSIF is equivalent to ulSIF when alpha=0.0")
    parser.add_argument("--fold",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Number of cross validation folds")
    parser.add_argument("--exe",
                        type=int,
                        default=30,
                        nargs="?",
                        help="Number of executions")
    parser.add_argument("--plot",
                        type=bool,
                        default=False,
                        nargs="?",
                        help="Plot images")
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
    n = args.n
    k = args.k
    alpha = args.alpha # equivalent to ulSIF when alpha=0.0
    fold = args.fold
    exe = args.exe
    RESULTS_DIR = "/" + args.results_dir + "/" + args.results_folder + "/" + args.train_or_test + "/"
    # check actions input shape
    print("Input action shape: " + str(X.shape))
    # repeat exe iterations
    results_1 = np.zeros((4,10,30))
    results_5 = np.zeros((4,10,30))
    results_10 = np.zeros((4,10,30))
    detection_delays = np.zeros((10,30))
    for e in range(0, exe):
        # calculate scores using RulSIF
        scores_1 = change_detection(X, action_index.values(), action_index_location, 
            timestamps, days, hours, seconds_past_midnight,
            n, k, alpha, fold)
        scores_2 = change_detection(np.flip(X), action_index.values(), action_index_location, 
            np.flip(timestamps), np.flip(days), np.flip(hours), np.flip(seconds_past_midnight),
            n, k, alpha, fold)
        scores_1 = np.array(scores_1)
        scores_2 = np.flip(np.array(scores_2))
        scores_sum = np.sum(np.array([scores_1, scores_2]), axis=0)
        scores_sum = np.concatenate((np.zeros(2*n-2+k), scores_sum))
        min_max_scaler = preprocessing.MinMaxScaler()
        scores_sum_norm = min_max_scaler.fit_transform(scores_sum.reshape(-1, 1))
        # plot in pieces
        if args.plot:
            points = 50
            number_of_plots = int(ceil(len(y) / points))
            print("Number of plots: " + str(number_of_plots))
            for i in range(0, number_of_plots):
                offset = i * points
                y_sub = y[offset:offset+points]
                scores_sum_norm_sub = scores_sum_norm[offset:offset+points]
                if offset + points < len(scores_sum_norm):
                    t = list(range(offset, offset+points))
                else:
                    t = list(range(offset, len(scores_sum_norm)))
                fig, ax = plt.subplots()
                for j in range(0, len(y_sub)):
                    if y_sub[j] == 1:
                        ax.axvline(offset+j, 0, max(scores_sum_norm), c='k')
                ax.plot(t, scores_sum_norm_sub, color='r')
                ax.set(xlabel='x', ylabel='score')
                print("Saving plot with number: " + str(i))
                fig.savefig(RESULTS_DIR + "scores_and_cp_ts_" + str(i) + ".png")
        # prepare change detection with offset using different threshold values
        counter_threshold = 0
        for threshold in [x * 0.1 for x in range(0, 10)]:
            cf_matrix_1 = get_conf_matrix_with_offset_strategy(scores_sum_norm, y, timestamps, threshold, 1)
            cf_matrix_5 = get_conf_matrix_with_offset_strategy(scores_sum_norm, y, timestamps, threshold, 5)
            cf_matrix_10 = get_conf_matrix_with_offset_strategy(scores_sum_norm, y, timestamps, threshold, 10)
            # TPR, TNR, FPR, G-MEAN for exact change point detection
            TN, FP, FN, TP = cf_matrix_1.ravel()
            TPR = TP/(TP+FN)
            TNR = TN/(TN+FP)
            FPR = FP/(FP+TN)
            G_MEAN = sqrt(TPR * TNR)
            results_1[0][counter_threshold][e] = TPR
            results_1[1][counter_threshold][e] = TNR
            results_1[2][counter_threshold][e] = FPR
            results_1[3][counter_threshold][e] = G_MEAN
            # TPR, TNR, FPR, G-MEAN for 5 second offset change point detection
            TN, FP, FN, TP = cf_matrix_5.ravel()
            TPR = TP/(TP+FN)
            TNR = TN/(TN+FP)
            FPR = FP/(FP+TN)
            G_MEAN = sqrt(TPR * TNR)
            results_5[0][counter_threshold][e] = TPR
            results_5[1][counter_threshold][e] = TNR
            results_5[2][counter_threshold][e] = FPR
            results_5[3][counter_threshold][e] = G_MEAN
            # TPR, TNR, FPR, G-MEAN for 10 second offset change point detection
            TN, FP, FN, TP = cf_matrix_10.ravel()
            TPR = TP/(TP+FN)
            TNR = TN/(TN+FP)
            FPR = FP/(FP+TN)
            G_MEAN = sqrt(TPR * TNR)
            results_10[0][counter_threshold][e] = TPR
            results_10[1][counter_threshold][e] = TNR
            results_10[2][counter_threshold][e] = FPR
            results_10[3][counter_threshold][e] = G_MEAN
            # detection delay
            detection_delay = get_detection_delay(scores_sum_norm, y, timestamps, threshold)
            detection_delays[counter_threshold][e] = detection_delay
            counter_threshold += 1
    # save population of results to file
    for threshold_num in range(0, 10):
        save_pop_results_to_file(results_1, 1, threshold_num)
        save_pop_results_to_file(results_5, 5, threshold_num)
        save_pop_results_to_file(results_10, 10, threshold_num)
        np.savetxt(RESULTS_DIR + 'detection_delays/' + str(threshold_num) + '_detection_delay' + '.csv', detection_delays[threshold_num])

if __name__ == "__main__":
    main(sys.argv)