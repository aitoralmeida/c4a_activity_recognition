import numpy as np

def extract_features_from_sensors(actions, unique_actions, all_actions, position, timestamps, all_timestamps):
    features_from_sensors = []
    
    # count of events for each sensor in window
    for action in unique_actions:
        counter = 0
        for action_fired in actions:
            if action == action_fired:
                counter += 1
        features_from_sensors.append(counter)

    return features_from_sensors