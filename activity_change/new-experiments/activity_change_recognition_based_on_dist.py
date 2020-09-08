import json
import sys

from gensim.models import Word2Vec
from scipy.spatial import distance

import h5py

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dot, Bidirectional, LSTM, Concatenate, Convolution2D, Dense, Dropout, Embedding, Flatten, GRU, Input, Lambda, MaxPooling2D, Multiply, Reshape
from keras.models import load_model, Model
from keras.preprocessing.text import Tokenizer
from keras_self_attention import SeqWeightedAttention

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import spatial

# Kasteren dataset
DIR = '/experiments/kasteren_dataset/'
# Dataset with vectors but without the action timestamps
DATASET_CSV = DIR + 'base_kasteren_reduced.csv'
# Word2Vec model
WORD2VEC_MODEL = DIR + 'actions_w_1.model'
# Number of input actions for the model
INPUT_ACTIONS = 5
# Number of elements in the action's embbeding vector
ACTION_EMBEDDING_LENGTH = 50
# List of unique activities in the dataset
UNIQUE_ACTIVITIES = DIR + 'unique_activities.json'
# List of unique actions in the dataset
UNIQUE_ACTIONS = DIR + 'unique_actions.json'
# Best model in the training
BEST_MODEL = '/results/best_model.hdf5'

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
    temp = ''
    actions = df_dataset['action']
    for i in range(1, len(actions)):
        temp = temp + str(actions[i]) + ' '
        if (i % 5 == 0):
            print(temp)
            temp = ''
        elif i == (len(actions)-1):
            print(temp)
    # get unique activities
    unique_activities = json.load(open(UNIQUE_ACTIVITIES, 'r'))
    total_activities = len(unique_activities)
    # get unique actions
    unique_actions = json.load(open(UNIQUE_ACTIONS, 'r'))
    total_actions = len(unique_actions)
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
    # check action embedding matrix
    print(embedding_action_matrix)
    # check distances from action i to action i +1 based on embeddings
    for i in range(0, len(X)-1):
        distance = 1 - spatial.distance.cosine(embedding_action_matrix[X[i]], embedding_action_matrix[X[i+1]])
        label = 'no' if (y[i+1]==0) else 'yes'
        for action, index in tokenizer_action.word_index.items():
            if index == X[i]:
                print("Action 1: " + action)
        for action, index in tokenizer_action.word_index.items():
            if index == X[i+1]:
                print("Action 2: " + action)
        print("Cosine distance is: " + str(distance))
        print("New Activity: " + label)

if __name__ == "__main__":
    main(sys.argv)
