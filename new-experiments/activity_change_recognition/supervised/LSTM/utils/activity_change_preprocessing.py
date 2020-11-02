import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from datetime import datetime as dt

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

def prepare_x_y_activity_change_with_input_actions_one_hot(df, input_actions):
    actions = df['action'].values
    activities = df['activity'].values
    dates = list(df.index.values)
    timestamps = list(map(get_unixtime, dates))

    tokenizer_action = Tokenizer(lower=False)
    tokenizer_action.fit_on_texts(actions.tolist())
    action_index = tokenizer_action.word_index
    
    tokenizer_activity = Tokenizer(lower=False)
    tokenizer_activity.fit_on_texts(activities.tolist())
    activity_index = tokenizer_activity.word_index  
    
    actions_by_index = []
    for action in actions:
        actions_by_index.append(action_index[action])
    
    activities_by_index = []
    for activity in activities:
        activities_by_index.append(activity_index[activity])
    
    last_action = len(actions) - 1
    X_actions = []
    X_activities = []
    y = []
    last_activity = None
    for i in range(last_action-input_actions):
        X_actions.append(actions_by_index[i:i+input_actions])
        X_activities.append(activities_by_index[i:i+input_actions])
        
        target_activity_change_onehot = np.zeros(2)
        if i == 0:
            target_activity_change_onehot[0] = 1.0
            last_activity = activities_by_index[i+input_actions-1]
        else:
            if last_activity == activities_by_index[i+input_actions-1]:
                target_activity_change_onehot[0] = 1.0
            else:
                target_activity_change_onehot[1] = 1.0
            last_activity = activities_by_index[i+input_actions-1]
        y.append(target_activity_change_onehot)

    X_actions = np.array(X_actions)
    X_activities = np.array(X_activities)
    y = np.array(y)
    timestamps = np.array(timestamps)
    
    return X_actions, X_activities, timestamps, y, tokenizer_action, tokenizer_activity   