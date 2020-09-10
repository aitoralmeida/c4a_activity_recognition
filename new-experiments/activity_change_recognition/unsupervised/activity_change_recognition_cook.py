import json
import sys

from gensim.models import Word2Vec
from scipy.spatial import distance

import h5py

from sklearn.metrics import classification_report

from keras.preprocessing.text import Tokenizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import spatial

# Kasteren dataset DIR
DIR = '../kasteren_house_a/'
# Kasteren dataset file
DATASET_CSV = DIR + 'base_kasteren_reduced.csv'
# Kasteren dataset with patterns
DATASET_CSV_WITH_PATTERNS = DIR + 'test_kasteren_removed.csv.annotated'

def prepare_x_y_activity_change(df):
    actions = df['action'].values
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
    
    return X, y, tokenizer_action

def main(argv):
    print(('*' * 20))
    print('Loading dataset...')
    sys.stdout.flush()
    # dataset of actions and activities
    DATASET = DATASET_CSV
    df_dataset = pd.read_csv(DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]
    # dataset with patterns extracted from cook's AD algorithm
    DATASET_WITH_PATTERNS = DATASET_CSV_WITH_PATTERNS
    df_dataset_with_patterns = pd.read_csv(DATASET_WITH_PATTERNS, parse_dates=[[0, 1]], header=None, index_col=0, sep='\t')
    df_dataset_with_patterns.columns = ['sensor', 'action', 'event', 'pattern']
    df_dataset_with_patterns.index.names = ["timestamp"]
    # check dataset struct
    print("### Dataset ###")
    print(df_dataset)
    print("### ### ###")
    # prepare dataset
    X, y, tokenizer_action = prepare_x_y_activity_change(df_dataset)
    # check prepared dataset struct
    print("### Actions ###")
    print(X)
    print("### ### ###")
    print("### Activity change ###")
    print(y)
    print("### ### ###")
    print("### Action Index ###")
    print(tokenizer_action.word_index)
    print("### ### ###")
    # detect activity change with cook's AD algorithm patterns
    counter = 0
    y_pred = []
    y_pred.append(0)
    last_pattern = df_dataset_with_patterns['pattern'][0]
    for i in range(0, len(X)-1):
        label = 'no' if (y[i+1]==0) else 'yes'
        predicted_label = 'yes' if (df_dataset_with_patterns['pattern'][i] != last_pattern) else 'no'
        if (predicted_label == 'yes'):
            y_pred.append(1)
        else:
            y_pred.append(0)
        last_pattern = df_dataset_with_patterns['pattern'][i]
    # print metrics
    print(classification_report(y, y_pred, target_names=['no', 'yes']))

if __name__ == "__main__":
    main(sys.argv)