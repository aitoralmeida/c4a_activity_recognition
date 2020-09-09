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
from keras_self_attention import SeqSelfAttention

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Kasteren dataset
DIR = '/experiments/kasteren_dataset/'
# Dataset with vectors but without the action timestamps
DATASET_CSV = DIR + 'base_kasteren_reduced.csv'
# Word2Vec model
WORD2VEC_MODEL = DIR + 'actions.model'
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
# Predict next activity label or only change
PREDICT_ACTIVITY_LABEL = True

def prepare_x_y_activity_label(df, unique_activities):
    actions = df['action'].values
    activities = df['activity'].values
    print(('total actions', len(actions)))
    print(('total activities', len(activities)))

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
    for i in range(last_action-INPUT_ACTIONS):
        X_actions.append(actions_by_index[i:i+INPUT_ACTIONS])
        X_activities.append(activities_by_index[i:i+INPUT_ACTIONS])
        target_activity = ''.join(i for i in activities[i+INPUT_ACTIONS-1] if not i.isdigit())
        target_activity_onehot = np.zeros(len(unique_activities))
        target_activity_onehot[unique_activities.index(target_activity)] = 1.0
        y.append(target_activity_onehot)
    
    return X_actions, X_activities, y, tokenizer_action, tokenizer_activity   

def prepare_x_y_activity_change(df):
    actions = df['action'].values
    activities = df['activity'].values
    print(('total actions', len(actions)))
    print(('total activities', len(activities)))

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
    for i in range(last_action-INPUT_ACTIONS):
        X_actions.append(actions_by_index[i:i+INPUT_ACTIONS])
        X_activities.append(activities_by_index[i:i+INPUT_ACTIONS])
        
        target_activity_change_onehot = np.zeros(2)
        if i == 0:
            target_activity_change_onehot[0] = 1.0
            last_activity = activities_by_index[i+INPUT_ACTIONS-1]
        else:
            if last_activity == activities_by_index[i+INPUT_ACTIONS-1]:
                target_activity_change_onehot[0] = 1.0
            else:
                target_activity_change_onehot[1] = 1.0
            last_activity = activities_by_index[i+INPUT_ACTIONS-1]
        y.append(target_activity_change_onehot)
    
    return X_actions, X_activities, y, tokenizer_action, tokenizer_activity   
    
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

def create_activity_embedding_matrix(tokenizer, df_dataset, unknown_actions, model):
    activity_index = tokenizer.word_index
    aux_embedding_activity_action_dict = {}
    
    for activity, i in list(activity_index.items()):
        aux_embedding_activity_action_dict[activity] = []
    for index, row in df_dataset.iterrows():
        if row['action'] not in unknown_actions.keys():
            aux_embedding_activity_action_dict[row['activity']].append(model[row['action']])
    
    embedding_activity_action_dict = {}
    for k, v in aux_embedding_activity_action_dict.items():
        array_avg = np.zeros(50)
        for array in v:
            array_avg = np.array(array) + array_avg
        array_avg = [x / len(v) for x in array_avg]
        embedding_activity_action_dict[k] = array_avg
    
    embedding_matrix = np.zeros((len(activity_index) + 1, ACTION_EMBEDDING_LENGTH))
    counter = 0
    for k, v in embedding_activity_action_dict.items():
        embedding_matrix[counter] = v

    return embedding_matrix

def main(argv):
    print(('*' * 20))
    print('Loading dataset...')
    sys.stdout.flush()
    # dataset of actions and activities
    DATASET = DATASET_CSV
    df_dataset = pd.read_csv(DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]    
    # recover all the actions in order
    actions = df_dataset['action'].values
    # recover all the activity in order
    activities = df_dataset['activity'].values
    # get unique activities
    unique_activities = json.load(open(UNIQUE_ACTIVITIES, 'r'))
    total_activities = len(unique_activities)
    # get unique actions
    unique_actions = json.load(open(UNIQUE_ACTIONS, 'r'))
    total_actions = len(unique_actions)
    # prepare dataset
    if PREDICT_ACTIVITY_LABEL:
        X_actions, X_activities, y, tokenizer_action, tokenizer_activity = prepare_x_y_activity_label(df_dataset, unique_activities)
    else:
        X_actions, X_activities, y, tokenizer_action, tokenizer_activity = prepare_x_y_activity_change(df_dataset)    
    # create the  action embedding matrix for the embedding layer initialization
    embedding_action_matrix, model, unknown_actions = create_action_embedding_matrix(tokenizer_action)
    # create the  action embedding matrix for the embedding layer initialization
    embedding_activity_matrix = create_activity_embedding_matrix(tokenizer_activity, df_dataset, unknown_actions, model)
    # divide the examples in training and validation
    total_examples = len(X_actions)
    test_per = 0.2
    limit = int(test_per * total_examples)
    X_actions_train = X_actions[limit:]
    X_activities_train = X_activities[limit:]
    X_actions_test = X_actions[:limit]
    X_activities_test = X_activities[:limit]
    y_train = y[limit:]
    y_test = y[:limit]
    print(('Different actions:', len(actions)))
    print(('Total examples:', total_examples))
    print(('Train examples:', len(X_actions_train), len(y_train))) 
    print(('Test examples:', len(X_actions_test), len(y_test)))
    sys.stdout.flush()  
    X_actions_train = np.array(X_actions_train)
    X_activities_train = np.array(X_activities_train)
    y_train = np.array(y_train)
    X_actions_test = np.array(X_actions_test)
    X_activities_test = np.array(X_activities_test)
    y_test = np.array(y_test)
    print('Shape (X,y):')
    print((X_actions_train.shape))
    print((X_activities_train.shape))
    print((y_train.shape))

    # some checks #################
    print("!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!")
    for i in range(0, 15):
        print(X_actions_train[i])
        print(X_activities_train[i])
        print(y_train[i])
    print("!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!")
    ###############################

    print(('*' * 20))
    print('Building model...')
    sys.stdout.flush()

    # input pipeline
    BATCH_SIZE = 128
    input_actions = Input(shape=(INPUT_ACTIONS,), dtype='int32', name='input_actions', batch_size=BATCH_SIZE)
    # embeddings
    embedding_actions = Embedding(input_dim=embedding_action_matrix.shape[0], output_dim=embedding_action_matrix.shape[1], weights=[embedding_action_matrix], input_length=INPUT_ACTIONS, trainable=True, name='embedding_actions')(input_actions)
    # bidirectional LSTM self-attention
    bidirectional_LSTM = Bidirectional(LSTM(units=128, return_sequences=True, input_shape=(INPUT_ACTIONS, ACTION_EMBEDDING_LENGTH), dropout=0.2, recurrent_dropout=0.2))(embedding_actions)
    self_attention = SeqSelfAttention(attention_activation='sigmoid')(bidirectional_LSTM)
    # denses
    flatten = Flatten()(self_attention)
    dense_1 = Dense(256, activation='relu', name='dense_1')(flatten)
    drop_1 = Dropout(0.8, name='drop_1')(dense_1)
    # prediction
    model = None
    if PREDICT_ACTIVITY_LABEL:
        # predict activity label
        output_activities = Dense(total_activities, activation='softmax', name='main_output')(drop_1)
        # build
        model = Model(input_actions, output_activities)
        # compile
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    else:
        # predict change of activity
        output_activities = Dense(2, activation='sigmoid', name='main_output')(drop_1)
        # build
        model = Model(input_actions, output_activities)
        # compile
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    print((model.summary()))
    sys.stdout.flush()
    # model train
    print(('*' * 20))
    print('Training model...')    
    sys.stdout.flush()
    checkpoint = ModelCheckpoint(BEST_MODEL, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    history = model.fit(X_actions_train, y_train, batch_size=BATCH_SIZE, epochs=1000, validation_data=(X_actions_test, y_test), shuffle=True, callbacks=[checkpoint, early_stopping])
    # model eval
    print(('*' * 20))
    print('Evaluating best model...')
    sys.stdout.flush()    
    model = load_model(BEST_MODEL)
    metrics = model.evaluate(X_actions_test, y_test, batch_size=BATCH_SIZE)
    print(metrics)

if __name__ == "__main__":
    main(sys.argv)
