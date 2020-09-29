import json
import sys
import numpy as np
import pandas as pd
import ruptures as rpt
import itertools
import operator

from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from scipy import spatial
from datetime import datetime
from scipy.stats import entropy

# Kasteren dataset DIR
DIR = '../../kasteren_house_a/'
# Kasteren dataset file
DATASET_CSV = DIR + 'base_kasteren_reduced.csv'
# List of unique actions in the dataset
UNIQUE_ACTIONS = DIR + 'unique_actions.json'
# Context information for the actions in the dataset
CONTEXT_OF_ACTIONS = DIR + 'context_model.json'

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

##################################################################################################################
# Feature extraction approach of Real-Time Change Point Detection with Application to Smart Home Time Series Data
# https://ieeexplore.ieee.org/document/8395405
# START
##################################################################################################################

def convert_epoch_time_to_hour_of_day(epoch_time_in_seconds):
    d = dt.fromtimestamp(epoch_time_in_seconds)
    return d.strftime('%H')

def convert_epoch_time_to_day_of_the_week(epoch_time_in_seconds):
    d = dt.fromtimestamp(epoch_time_in_seconds)
    return d.strftime('%A')

def get_seconds_past_midnight_from_epoch(epoch_time_in_seconds):
    date = dt.fromtimestamp(epoch_time_in_seconds)
    seconds_since_midnight = (date - date.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    return seconds_since_midnight

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
    if previous_actions_1 != None:
        most_frequent_sensor_1 = most_common(previous_actions_1)
    else:
        most_frequent_sensor_1 = 0 # non-existing sensor index
    if previous_actions_2 != None:
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
    
    return X, timestamps, hours, days, seconds_past_midnight, y, tokenizer_action

def main(argv):
    np. set_printoptions(suppress=True)
    print(('*' * 20))
    print('Loading dataset...')
    sys.stdout.flush()
    # dataset of actions and activities
    DATASET = DATASET_CSV
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
    print("### Dataset ###")
    print(df_dataset)
    print("### ### ### ###")
    # prepare dataset
    X, timestamps, hours, days, seconds_past_midnight, y, tokenizer_action = prepare_x_y_activity_change(df_dataset)
    # transform action:location dict struct to action_index:location struct
    action_index = tokenizer_action.word_index
    action_index_location = {}
    for key, value in action_index.items():
        action_index_location[value] = action_location[key]
    # check action_index:location struct
    print(action_index_location)
    # check prepared dataset struct
    print("### Actions ###")
    print(X)
    print("### ### ### ###")
    print("### Activity change ###")
    print(y)
    # perform window-based feature extraction based on Aminikhanghahi et al. paper
    feature_vectors = []
    previous_actions_1 = None
    previous_actions_2 = None
    for i in range(len(X)):
        actions_of_window, timestamps_of_window = extract_actions_from_window(X, timestamps, i, 30)
        feature_vector = []
        # time features
        feature_vector.append(int(hours[i]))
        feature_vector.append(day_to_int(days[i]))
        feature_vector.append(seconds_past_midnight[i])
        # window features
        window_features = extract_features_from_window(actions_of_window, previous_actions_1, previous_actions_2, action_index_location, timestamps_of_window)
        feature_vector.extend(window_features)
        # sensor features
        sensor_features = extract_features_from_sensors(actions_of_window, action_index.values(), X, i, timestamps_of_window, timestamps)
        feature_vector.extend(sensor_features)
        # update previous actions
        previous_actions_2 = previous_actions_1
        previous_actions_1 = actions_of_window
        # append feature vector to list
        feature_vectors.append(feature_vector)
    # check first feature vector struct
    print(feature_vectors[0])
    # normalize min max scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    features_vectors_norm = min_max_scaler.fit_transform(feature_vectors)
    # check first feature vector struct normalized
    print(features_vectors_norm[0])
    # check shape
    print(features_vectors_norm.shape)
    # change point detection
    model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
    algo = rpt.Window(width=2, min_size=1, model=model).fit(features_vectors_norm)
    result = algo.predict(epsilon=0.3) # threshold = epsilon
    print(result)
    # prepare results for eval
    y_pred = np.zeros(len(y))
    for change_point_index in result:
        y_pred[change_point_index-1] = 1
    # eval
    print(classification_report(y, y_pred, target_names=['no', 'yes']))

if __name__ == "__main__":
    main(sys.argv)