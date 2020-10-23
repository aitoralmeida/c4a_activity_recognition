import sys
import numpy as np

sys.path.append('../densratio')
from core import densratio

from math import ceil

from feature_extraction_RuLSIF_2 import *

##################################################################################################################
# R_ULSIF based CPD algorithm translation from http://allmodelsarewrong.net/software.html
# START
##################################################################################################################  

def sliding_window_with_features(actions, unique_actions, locations, timestamps, days, hours, seconds_past_midnight, window_size, step):
    windows = None
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
            # sensor features (only action count)
            sensor_features = extract_features_from_sensors(window_actions, unique_actions, actions, i, window_timestamps, timestamps)
            feature_vector.extend(sensor_features)
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

        if t % 20 == 0:
            print(t)

        t += 1
    
    print("Num of scores: " + str(len(scores)))

    return scores

##################################################################################################################
# R_ULSIF based CPD algorithm translation from MatLab code http://allmodelsarewrong.net/software.html
# END
################################################################################################################## 