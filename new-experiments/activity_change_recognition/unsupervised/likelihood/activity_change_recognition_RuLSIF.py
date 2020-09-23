import json
import sys
import numpy as np
import pandas as pd
import ruptures as rpt
import itertools
import operator
import csv
from densratio import densratio

from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from scipy import spatial
from scipy.stats import entropy, norm
from datetime import datetime as dt

from pylab import *
from scipy import linalg
from scipy.stats import norm

# Kasteren dataset DIR
DIR = '../../kasteren_house_a/'
# Kasteren dataset file
DATASET_CSV = DIR + 'base_kasteren_reduced.csv'
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

##################################################################################################################
# R_ULSIF based CPD algorithm translation from http://allmodelsarewrong.net/software.html
# START
##################################################################################################################  

def sliding_window(X, window_size, step):
    (num_dims, num_samples) = X.shape

    windows = np.zeros((num_dims * window_size * step, num_samples + 1 - window_size * step))
    print("Allocated window struct shape: " + str(windows.shape))
    for i in range(0, num_samples):
        if (i % step == 0):
            offset = window_size * step
            if i + offset > num_samples:
                break
            window = X[:,i:(i+offset)]
            windows[:,int(ceil(i/step))] = window[:]
    windows = np.array(windows)

    return windows
            
def change_detection(X, n, k, alpha, fold):
    scores = []
    
    windows = sliding_window(X, k, 1)
    num_samples = windows.shape[1]
    print("Num window samples in change detection: " + str(num_samples))
    t = n

    while((t+n) <= num_samples):
        y = windows[:,(t-n):(n+t)]
        y = y / np.std(y)
        # print("Y shape: " + str(y.shape))
        y_ref = y[:,0:n]
        # print("Y ref shape: " + str(y_ref.shape))
        y_test = y[:,n:]
        # print("Y test shape: " + str(y_test.shape))

        #(PE, w, score) = R_ULSIF(y_test, y_ref, [], alpha, sigma_list(y_test,y_ref),lambda_list(), y_test.shape[1], 5)
        densratio_obj = densratio(y_test, y_ref, alpha=alpha)

        #print("Score: " + str(PE))
        #scores.append(PE)
        scores.append(densratio_obj.alpha_PE)

        if mod(t,20) == 0:
            print(t)

        t += 1
    
    print("Num of scores: " + str(len(scores)))

    return scores

##################################################################################################################
# R_ULSIF based CPD algorithm translation from MatLab code http://allmodelsarewrong.net/software.html
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
    np.set_printoptions(threshold=sys.maxsize)
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
    # write feature vectors to CSV
    with open('/results/kasteren_ts_feature_vectors.csv', mode='w') as kasteren_ts_feature_vectors:
        for i in range(0, len(feature_vectors)):
            kasteren_ts_feature_vectors_writer = csv.writer(kasteren_ts_feature_vectors, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = []
            row.append(timestamps[i])
            row.append(y[i])
            for j in range(0, len(feature_vectors[i])):
                row.append(feature_vectors[i][j])
            kasteren_ts_feature_vectors_writer.writerow(row)
    # write feature vectors norm to CSV
    with open('/results/kasteren_ts_feature_vectors_norm.csv', mode='w') as kasteren_ts_feature_vectors_norm:
        for i in range(0, len(features_vectors_norm)):
            kasteren_ts_feature_vectors_norm_writer = csv.writer(kasteren_ts_feature_vectors_norm, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = []
            row.append(timestamps[i])
            row.append(y[i])
            for j in range(0, len(features_vectors_norm[i])):
                row.append(features_vectors_norm[i][j])
            kasteren_ts_feature_vectors_norm_writer.writerow(row)
    # change point detection
    n = 50
    k = 10
    alpha = 0.01
    fold = 5
    # testing with a simple timeseries
    prueba = [112310,112530,113710,115770,116930,115560,113110,111520,111680,113200,114690,114830,114970,116130,116640,114740,112010,110380,110580,112080,114220,115790,116290,115480,113540,111040,109720,110470,112810,115190,116130,114950,113860,114440,114810,113030,111260,111790,115300,120140,121970,118180,113710,113570,116250,117990,118350,117850,117040,116440,116780,117630,117450,115020,111570,108870,108270,109690,111570,112240,112470,113280,114690,115930,116620,116350,115640,115010,114740,114700,115760,117900,118940,117300,116200,118050,119900,118390,115560,114140,114670,116070,116790,115670,114330,114320,115650,116990,116650,114010,112130,113470,116530,118380,118690,117840,116020,113700,112700,113740,115440,115900,115150,113910,113190,113310,114210,114970,114290,111840,110240,111500,115260,119200,119800,115680,112660,115270,118580,116830,113090,111820,113090,115000,116370,116180,115020,113930,112890,111680,112890,117160,118880,114070,109400,111220,115630,116710,115740,115300,115090,114240,113920,114600,114520,112360,110970,112500,114530,114210,113170,113510,115300,117300,118240,117240,115340,113920,114020,115410,116770,116680,115940,115530,115460,115130,114420,113270,112890,113770,114650,114080,113060,112710,113160,113750,113970,113380,113070,113780,114480,114000,113990,115490,116190,114120,112070,112710,115010,116550,117010,116480,114760,112190,110990,112380,114920,116390,117050,117340,116850,115180,112730,110400,111310,116320,120490,119290,115780,114260,114910,116100,116940,116680,117010,118680,117550,111090,106210,108730,114200,116230,115450,114430,114630,116190,117130,115570,114340,115720,116660,114140,111470,111980,114100,115060,115250,115410,115400,114770,113920,113140,112900,113180,114370,116040,116000,113130,111990,115610,119070,117310,114180,114400,116450,117310,117930,119100,118780,115570,112470,112170,113620,114570,115070,115480,116110,116740,116510,114840,113560,114120,116050,117720,118120,116850,115490,115300,115690,115370,115190,115690,115810,114540,113560,114180,115290,115380,115290,115850,117010,117940,117520,115300,113940,115330,117390,117440,116950,117510,117950,116850,115250,114440,115640,118490,120410,119140,116310,114240,113600,113780,114050,113710,113450,113890,115310,116950,116840,113940,111590,112570,114880,115520,115400,115820,116390,116240,115760,115340,115280,115590,117260,120360,122710,122600,123250,126930,129610,127480,124730,125610,127540,127000,126170,127340,129020,129210,128780,128890,130270,132320,133010,130820,127710,126060,126870,129280,131370,131260,129430,127320,126330,126860,129020,131660,132230,129460,127520,129530,131900,130480,128010,127900,129520,130850,131510,131340,130970,130730,129830,127690,126700,128440,131160,132310,132490,132580,131880,129690,128270,129000,129350,127020,125750,128370,130790,128820,125800,125670,127600,129090,130140,130840,130120,127660,126900,129700,131920,129480,126130,125900,126130,123680,122340,125170,128740,129050,128270,129060,130470,130600,129940,129150,128290,127200,126770,127210,127640,127080,126660,127310,128400,128800,129460,130880,131390,129670,128760,130780,133000,132180,130010,128790,128470,128110,127860,127720,128430,129900,130420,128650,126960,127510,129190,129950,130670,131930,131970,129400,126660,126030,126830,127380,128210,129660,130660,130170,129810,130690,131080,129340,128590,131010,133120,131310,129040,129720,129990,126640,124940,128910,133360,132610,129540,128460,129830,131960,133080,131780,129540,128210,127840,127580,127840,128530,128770,127800,126760,126620,127150,127480,128070,128930,129020,127450,126150,126590,126940,125540,125860,130200,134410,134120,131820,130740,130750,130260,128230,124670,123510,127110,131560,132340,131100,130630,131310,132100,132090,130550,128690,127750,128120,129140,130210,130570,130250,129480,129440,130400,130810,129220,128090,129320,130030,127650,126610,130210,134060,133220,129920,127830,128140,130140,131970,131860,131730,133260,134230,132320,130640,131690,132020,128430,125770,128050,130330,127680,125040,127410,131800,133440,132820,131500,128960,125180,124400,128720,132840,131740,130320,133230,135640,132350,127450,125950,127480,129510,130990,131240,130580,129690,129690,130720,131630,131180,130920,132010,132960,131970,130550,130050,129090,126460,124730,125890,128980,131500,131720,129260,126980,127360,129590,131420,132460,132670,132300,131480,130220,128460,127300,127440,128650,129920,131280,132300,131430,127970,125430,126410,128620,128850,128720,129890,130310,128140,126340,127380,129630,130450,130400,130490,130680,130530,130620,130930,130620,128990,127060,126040,127060,129890,132660,133250,132660,131950,129720,125120,122010,123350,127330,130530,132530,133530,133070,130990,129200,128820,128750,127540,126790,127790,129190,129330,129280,130020,131540,132820,132290,129240,127030,128130,129790,128830,128210,130530,132520,130830,128720,129340,130510,129230,126830,125170,124680,124640,124140,122340,120170,118700,118470,119180,120210,120790,122120,124550,125250,122190,119090,119360,121700,122990,121710,117910,114970,115410,116780,116020,115860,118700,121260,120170,118820,120280,121310,118550,115750,116490,118010,116990,116050,117670,119880,119950,118210,115980,115530,117670,120000,119920,119370,120400,121880,121890,121130,120510,119840,118700,118570,119930,120450,118220,116180,117080,119260,120070,120790,122480,123010,120600,118120,118170,119390,119490,119450,120260,120870,120150,119430,119770,120640,120800,120100,118720,117320,116440,117180,119400,121090,120510,119760,120740,121760,120840,119640,119930,121860,124160,124710,122290,119560,119040,120050,120610,120130,118850,118500,120140,122910,124810,124610,122180,119520,118550,119520,121240,122870,123390,122190,119420,117030,116400,117030,117560,118240,119260,120090,120070,120090,120610,120590,119150,118340,119490,121100,120920,118660,115270,113760,115710,119040,120730,120440,119240,119600,122540,125090,124200,122290,122110,122750,122270,120880,119260,118530,119030,119620,119030,118160,118060,119530,122040,123150,121130,119190,120080,121310,119670,117550,117450,116850,113730,113520,119660,125670,124920,120860,118790,119180,120290,120750,119630,119110,120840,122360,121100,119630,120420,121910,121700,120740,120180,119300,117420,116680,118320,120150,119780,118860,119240,120410,121050,121180,121040,121500,122730,123650,122970,121460,120130,119030,117800,117840,119450,119630,116360,113950,116020,119940,121510,120770,119120,117700,117180,118210,120290,121850,121650,121850,123870,124560,121360,118290,119060,122180,124190,122850,118200,115040,117010,121260,123200,122650,121070,119490,118600,119570,121970,123190,121320,118940,118810,120370,121630,122330,122360,121210,118910,118060,120060,122420,122420,122320,124210,125900,124850,122680,121450,121370,121640,122030,122100,122040,121760,120970,119300,117410,115870,115980,117730,118700,117090,115720,117240,120600,123140,123630,121870,119350,117720,118060,119880,121540,121420,120190,119010,118900,119760,120320,119420,118450,118740,120450,122400,122300,119040,116730,118600,121250,120650,119940,122120,124100,122430,119530,118360,118730,119150,119470,119490,119240,118820,118700,119010,120100,121610,122850,122950,121830,119840,118200,117660,118200,119000,119580,119590,119230,118730,119230,120790,121010,118280,116500,118740,122120,122600,120860,118890,117460,116760,118180,121410,122900,120150,117210,117950,120250,120660,120180,120290,120820,120880,120340,119050,117690,116720,115840,114630,114340,115640,117410,118180,118540,119080,119920,120460,119700,117250,115880,117500,119870,120120,120070,121400,121400,117920,115170,116680,119620,120060,119250,119140,119320,118930,118680,119060,120500,122430,123290,121900,119650,118270,118580,119980,121330,121310,119610,116870,115850,117870,119870,118730,117410,118780,120170,118480,116490,116820,117050,114720,113150,115260,118620,119940,120520,121810,122170,120080,117910,117760,119020,119900,119970,119230,117930,116570,116870,119090,120770,119630,117050,115310,116420,120100,122480,120420,117430,117360,118850,119090,118490,118020,117900,118040,119240,121340,122410,120930,119420,119950,120490,118700,116760,116870,118490,119910,120170,119030,118730,120580,122380,121580,119610,118480,118630,119410,120410,120950,120920,120440,120170,120330,120840,121060,120840,120060,119280,118750,118870,119330,118870,116800,115910,118040,120740,121010,119870,119170,119870,121540,122850,122380,120400,118080,117720,120010,122550,122480,120150,117240,115940,117090,119150,119920,119410,118350,118060,118940,118970,116700,115920,119260,123160,123170,120110,116560,114300,114020,116470,120530,122260,119340,116990,119560,122440,120450,118130,119980,122130,120120,117170,116930,118150,118380,118190,118430,118820,118840,119720,121710,122060,118910,116310,117640,120190,120120,118020,115870,115660,117870,120190,120050,118980,118820,119030,118370,117210,116120,116170,117550,118920,118890,118460,118780,120000,121280,122140,121860,119750,116020,113460,113950,116210,117850,118300,117820,118090,119790,120400,117830,115570,116780,119420,120280,120010,119760,119140,117600,116270,115900,116780,118190,118670,117360,116150,116690,119080,121750,122010,118580,115560,116420,118750,118830,117610,116800,116330,115690,115700,116730,118110,118980,120430,122830,123970,122040,119470,118680,119120,119080,118230,116790,116060,116830,118270,119030,120010,121710,122840,122070,120690,119920,119770,119420,118030,115410,114400,116620,118650,117200,115180,115860,118710,121320,121760,119270,116830,117030,118370,118330,117650,117420,117530,117390,116880,116040,116230,117960,120230,121260,120230,117410,115080,115020,116940,119180,121000,121760,121000,118860,117370,117720,118730,118590,118330,118860,118960,117430,116380,117300,118740,118720,117630,116520,117180,119890,121240,118640,116740,119380,122210,120400,117160,116480,117680,118450,118190,116860,115940,116490,116910,115680,115930,119570,122440,120540,117270,116660,117690,117860,117900,118420,117930,115540,114350,116750,120580,122910,125710,130470,133370,131450,130070,133670,137360,135870,133620,135300,137870,137200,135330,134780,134630,133520,133380,135400,137960,138810,137960,136190,135210,135710,136190,135050,133970,134500,136410,138100,137230,133040,130510,133250,137220,137380,135770,135480,136120,136250,136000,135440,135570,136600,136490,133890,132950,136490,140110,139010,136340,135930,135950,133980,133000,135180,137820,137680,135680,133690,133400,134930,135780,133910,132700,134930,138080,138570,136750,134220,133310,134730,135840,134080,132010,132440,134870,137180,138220,137390,135900,134980,134720,134480,134740,135560,136530,137020,137310,137510,137750,137720,137080,135640,134700,134960,135930,136330,135520,133540,132480,133570,135310,135510,134580,133510,133040,133220,134150,135220,135610,134910,134920,136750,138780,138940,138400,138320,137290,133890,130360,128430,126450,122690,119280,118290,120010,122810,123410,120040,117200,118750,122380,123980,122990,120500,119280,120780,122320,120840,118620,118330,118410,117050,117420,121430,124320,121910,118730,119380,121270,120500,118350,116990,117090,118180,119620,120310,119620,117720,116710,117810,119850,120770,119850,117480,116270,117710,119630,119360,118010,117300,118010,119620,120000,117800,116930,120150,123220,121670,119090,119390,120090,118140,116020,116360,118820,121430,122170,120010,118150,119190,119960,117340,116440,120940,124170,120050,114920,115610,118910,119660,119500,120740,122100,121920,121650,122270,122320,120360,118130,117380,118310,119840,120490,119390,118700,120070,121870,121800,120760,120040,119310,118060,117840,119310,120900,120720,119010,116900,115980,116870,119050,120940,121390,120010,118500,118180,118910,119430,118560,116080,115070,117690,121690,123920,125810,128710,130340,128710,126480,126370,127840,129050,130180,131240,130960,128710,127160,128190,130270,130930,130450,129700,129500,129910,130490,130310,129540,128570,127990,127990,129260,131460,132420,130650,129460,131360,133110,131280,129190,130050,131590,130710,128780,127690,127860,128550,128860,127930,126780,126320,126320,126000,125790,125930,126530,127210,127790,127940,127800,127510,128190,129910,130790,129170,127090,126660,127340,127700,128670,130600,131690,130410,128740,128610,129760,130720,131100,130570,129610,128680,127820,126810,127540,130360,131300,127560,124160,125800,130070,132520,132750,131610,130060,128910,128290,127800,129090,132370,133750,130410,126700,126920,129370,130520,130180,129120,128740,129690,130910,130850,130530,130880,131180,130410,129760,130010,130200,129200,127960,127410,127800,128270,127360,123980,119600,116040,113970,113110,113970,116150,117910,117750,117160,117410,117340,115570,113410,112430,113410,115830,118070,118500,118070,117890,116950,114420,113410,115830,118020,116490,114900,116790,119080,118080,116650,117740,119220,118460,117740,118970,119860,118050,115690,115080,116150,117340,117230,115140,112840,112190,113640,116150,118310,118860,118500,118090,117740,116960,115620,113760,112840,113550,114320,113470,112530,112960,113890,113910,113840,114410,115260,115540,115770,116080,115140,112350,111150,113840,117450,118280,117370,116720,116620,116560,116500,116150,116110,116500,115890,113320,112000,114000,116030,114650,112410,112400,114130,115590,115890,114740,113480,113250,114210,115290,114950,112570,110900,112210,115770,119050,119920,117590,114650,113700,114650,115870,116800,117040,116790,116360,116310,116560,117100,117440,116570,114250,113680,116710,119610,118550,116220,115870,116480,116000,114690,113320,113680,116150,117810,116150,113620,113130,115220,118180,118730,114960,110710,110040,112050,113930,115540,116830,116480,114010,112030,112520,114740,116540,116870,115440,113330,111820,111640,112560,114440,116450,117180,115840,114500,114780,115390,114580,113990,115030,116510,116800,116510,116490,117280,118470,118530,116380,114270,114160,115040,115110,115040,115580,116660,117540,117370,115620,113960,113710,113960,113450,113740,115910,118130,118370,118130,118830,119120,117550,116420,117370,118250,116630,114030,112490,112370,112980,113830,114290,114880,115900,117050,117580,117580,117110,116480,115780]
    prueba = np.array(prueba)
    prueba_reshaped = prueba.reshape(1,-1)
    print("Input shape: " + str(prueba.shape))
    scores_1 = change_detection(prueba_reshaped, n, k, alpha, fold)
    scores_2 = change_detection(np.flip(prueba_reshaped), n, k, alpha, fold)
    scores_1 = np.array(scores_1)
    scores_2 = np.flip(np.array(scores_2))
    scores_sum = np.sum(np.array([scores_1, scores_2]), axis=0)
    # plot result of testing
    t = list(range(0, len(prueba)))
    fig, ax = plt.subplots(2)
    ax[0].plot(t, prueba, color='b')
    ax[0].set(xlabel='x', ylabel='y')
    ax[1].plot(t, np.concatenate((np.zeros(2*n-2+k), scores_sum)), color='r')
    ax[1].set(xlabel='x', ylabel='score')
    fig.savefig("/results/combined_scores_original_ts.png")
    # feature vectors

if __name__ == "__main__":
    main(sys.argv)