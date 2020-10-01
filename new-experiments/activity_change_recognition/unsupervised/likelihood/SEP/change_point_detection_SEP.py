import sys
import numpy as np

sys.path.append('../densratio')
from core import densratio

from math import ceil

from feature_extraction_SEP import *

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

        scores.append(densratio_obj.alpha_SEP)

        t += 1
    
    print("Num of scores: " + str(len(scores)))

    return scores

##################################################################################################################
# R_ULSIF based CPD algorithm translation from MatLab code http://allmodelsarewrong.net/software.html
# Adapted for Kasteren dataset feature extraction based on Aminikhanghahi et al. paper
# END
################################################################################################################## 