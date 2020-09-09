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
# Word2Vec model
WORD2VEC_MODEL = DIR + 'actions_w1.model'
# Embedding size
ACTION_EMBEDDING_LENGTH = 50
# MIN DIST if cosine distance < MIN_DIST then change of activity
MIN_DIST = 0.95

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
    
def create_action_embedding_matrix(tokenizer):
    model = Word2Vec.load(WORD2VEC_MODEL)    
    action_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(action_index) + 1, ACTION_EMBEDDING_LENGTH))
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
    
    return embedding_matrix, model, unknown_actions

def main(argv):
    print(('*' * 20))
    print('Loading dataset...')
    sys.stdout.flush()
    # dataset of actions and activities
    DATASET = DATASET_CSV
    df_dataset = pd.read_csv(DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]
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
    # create the  action embedding matrix for the embedding layer initialization
    embedding_action_matrix, model, unknown_actions = create_action_embedding_matrix(tokenizer_action)
    # check distances from action i to action i+1 based on embeddings and detect activity change
    counter = 0
    y_pred = []
    discovered_activities = []
    discovered_activity_num = 0
    y_pred.append(0)
    discovered_activities.append('ACT' + str(discovered_activity_num))
    for i in range(0, len(X)-1):
        distance = 1 - spatial.distance.cosine(embedding_action_matrix[X[i]], embedding_action_matrix[X[i+1]])
        label = 'no' if (y[i+1]==0) else 'yes'
        predicted_label = 'yes' if (distance < MIN_DIST) else 'no'
        if (predicted_label == 'yes'):
            discovered_activity_num += 1
            y_pred.append(1)
        else:
            y_pred.append(0)
        discovered_activities.append('ACT' + str(discovered_activity_num))
    # print metrics
    print(classification_report(y, y_pred, target_names=['no', 'yes']))
    # prepare dataset to be saved
    df_dataset['discovered_activity'] = discovered_activities
    df_dataset = df_dataset.drop("activity", axis=1)
    # save dataset (requires further processing)
    df_dataset.to_csv('/results/test_kasteren_cosine_distance.csv.annotated', header=None, index=True, sep=' ', mode='a')

if __name__ == "__main__":
    main(sys.argv)
