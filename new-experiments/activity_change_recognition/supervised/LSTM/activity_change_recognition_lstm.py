import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dot, Bidirectional, LSTM, Concatenate, Convolution2D, Dense, Dropout, Embedding, Flatten, GRU, Input, Lambda, MaxPooling2D, Multiply, Reshape
from keras.models import load_model, Model
from keras.preprocessing.text import Tokenizer
from keras_self_attention import SeqWeightedAttention

import json
import sys
import numpy as np
import pandas as pd
import argparse

from sklearn import preprocessing
from scipy import spatial
from pylab import *
from math import sqrt

from utils.activity_change_preprocessing import *
from utils.activity_change_save_results import *
from utils.activity_change_evaluation import *

from embeddings_utils.create_embedding_matrix import *

from gensim.models import Word2Vec
import multiprocessing

def main(argv):
    np.set_printoptions(threshold=sys.maxsize)
    np. set_printoptions(suppress=True)
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="../../kasteren_house_a",
                        nargs="?",
                        help="Dataset dir")
    parser.add_argument("--dataset_file",
                        type=str,
                        default="kasterenA_groundtruth.csv",
                        nargs="?",
                        help="Dataset file")
    parser.add_argument("--results_dir",
                        type=str,
                        default='results/kasteren_house_a',
                        nargs="?",
                        help="Dir for results")
    parser.add_argument("--results_folder",
                        type=str,
                        default='LSTM',
                        nargs="?",
                        help="Folder for results")
    parser.add_argument("--train_or_test",
                        type=str,
                        default='train',
                        nargs="?",
                        help="Specify train or test data")
    parser.add_argument("--embedding_size",
                        type=int,
                        default=50,
                        nargs="?",
                        help="Embedding size for word2vec algorithm")
    parser.add_argument("--window_size",
                        type=int,
                        default=1,
                        nargs="?",
                        help="Window size for word2vec algorithm")
    parser.add_argument("--iterations",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Iterations for word2vec algorithm")
    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        nargs="?",
                        help="Length of batches")
    parser.add_argument("--units",
                        type=int,
                        default=128,
                        nargs="?",
                        help="Number of units of LSTM")
    parser.add_argument("--input_actions",
                        type=int,
                        default=2,
                        nargs="?",
                        help="Number of input actions")
    parser.add_argument("--epochs",
                        type=int,
                        default=1000,
                        nargs="?",
                        help="Number of epochs")
    parser.add_argument("--patience",
                        type=int,
                        default=250,
                        nargs="?",
                        help="Patience in number of epochs")
    parser.add_argument("--exe",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Number of executions")
    args = parser.parse_args()
    print('Loading dataset...')
    sys.stdout.flush()
    # dataset of actions and activities
    DATASET = args.dataset_dir + "/" + args.dataset_file.replace('.csv', '') + "_" + args.train_or_test + ".csv"
    df_dataset = pd.read_csv(DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]
    # list of unique actions in the dataset
    UNIQUE_ACTIONS = args.dataset_dir + "/" + 'unique_actions.json'
    # context information for the actions in the dataset
    CONTEXT_OF_ACTIONS = args.dataset_dir + "/" + 'context_model.json'
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
    print("Dataset")
    print(df_dataset)
    # prepare dataset
    INPUT_ACTIONS = args.input_actions
    X_actions, X_activities, timestamps, y, tokenizer_action, tokenizer_activity = prepare_x_y_activity_change_with_input_actions_one_hot(df_dataset, INPUT_ACTIONS)
    # transform action:location dict struct to action_index:location struct
    action_index = tokenizer_action.word_index
    action_index_location = {}
    for key, value in action_index.items():
        action_index_location[value] = action_location[key]
    # check action_index:location struct
    print(action_index_location)
    # check prepared dataset struct
    print("Actions")
    print(X_actions)
    print("Activity change")
    print(y)
    # change point detection
    window_size = args.window_size
    iterations = args.iterations
    exe = args.exe
    embedding_size = args.embedding_size
    BATCH_SIZE = args.batch_size
    UNITS = args.units
    EPOCHS = args.epochs
    PATIENCE = args.patience
    RESULTS_DIR = "/" + args.results_dir + "/" + args.results_folder + "_" + str(INPUT_ACTIONS) + "/window_" + str(window_size) + "_iterations_" + str(iterations) + "_embedding_size_" + str(embedding_size) + "/" + args.train_or_test + "/"
    BEST_MODEL = RESULTS_DIR + 'best_model'
    # create dirs for saving results
    create_dirs(RESULTS_DIR, word2vec=True)
    print('Created dirs at: ' + RESULTS_DIR)
    # check actions input shape
    print("Input action shape: " + str(X_actions.shape))
    # repeat exe iterations
    results_1 = np.zeros((4,10,exe))
    results_5 = np.zeros((4,10,exe))
    results_10 = np.zeros((4,10,exe))
    detection_delays = np.zeros((10,exe))
    models = []
    for e in range(0, exe):
        if args.train_or_test == "train":
            # input pipeline
            input_actions = Input(shape=(INPUT_ACTIONS,), dtype='int32', name='input_actions')
            # embeddings
            model = Word2Vec([df_dataset['action'].values.tolist()],
                    size=embedding_size, window=window_size, min_count=0, iter=iterations, seed=np.random.randint(1000000),
                    workers=multiprocessing.cpu_count())
            models.append(model)
            embedding_action_matrix, unknown_actions = create_action_embedding_matrix(tokenizer_action, model, embedding_size)
            embedding_actions = Embedding(input_dim=embedding_action_matrix.shape[0], output_dim=embedding_action_matrix.shape[1], weights=[embedding_action_matrix], input_length=INPUT_ACTIONS, trainable=True, name='embedding_actions')(input_actions)
            # bidirectional LSTM
            bidirectional_LSTM = Bidirectional(LSTM(units=UNITS, input_shape=(INPUT_ACTIONS, embedding_size), dropout=0.2, recurrent_dropout=0.2, recurrent_activation="sigmoid"))(embedding_actions)
            # denses
            dense_1 = Dense(256, activation='relu', name='dense_1')(bidirectional_LSTM)
            drop_1 = Dropout(0.8, name='drop_1')(dense_1)
            # predict change of activity
            output_activities = Dense(2, activation='sigmoid', name='main_output')(drop_1)
            # build
            model = Model(input_actions, output_activities)
            # compile
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
            print((model.summary()))
            sys.stdout.flush()
            # model train
            print('Training model...')    
            sys.stdout.flush()
            checkpoint = ModelCheckpoint(BEST_MODEL + "_" + str(e) + ".hdf5", monitor='loss', verbose=0, save_freq='epoch', save_best_only=True, save_weights_only=False, mode='auto')
            early_stopping = EarlyStopping(monitor='loss', patience=PATIENCE)
            history = model.fit(X_actions, y, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, callbacks=[checkpoint, early_stopping])
        elif args.train_or_test == "test":
            # best model in training phase
            BEST_MODEL = "/" + args.results_dir + "/" + args.results_folder + "_" + str(INPUT_ACTIONS) + "/window_" + str(window_size) + "_iterations_" + str(iterations) + "_embedding_size_" + str(embedding_size) + "/" + "train" + "/" + 'best_model'
        # model eval
        print('Evaluating best model...')
        sys.stdout.flush()
        model = load_model(BEST_MODEL + "_" + str(e) + ".hdf5")
        y_pred = model.predict(X_actions, batch_size=BATCH_SIZE).argmax(axis=-1)
        print(y_pred)
        print(y.argmax(axis=-1))
        # conf matrix
        cf_matrix_1 = get_conf_matrix_with_offset_strategy_from_predicted_labels(y_pred, y.argmax(axis=-1), timestamps, 1)
        cf_matrix_5 = get_conf_matrix_with_offset_strategy_from_predicted_labels(y_pred, y.argmax(axis=-1), timestamps, 5)
        cf_matrix_10 = get_conf_matrix_with_offset_strategy_from_predicted_labels(y_pred, y.argmax(axis=-1), timestamps, 10)
        # TPR, TNR, FPR, G-MEAN for exact change point detection
        TN, FP, FN, TP = cf_matrix_1.ravel()
        TPR = TP/(TP+FN)
        TNR = TN/(TN+FP)
        FPR = FP/(FP+TN)
        G_MEAN = sqrt(TPR * TNR)
        results_1[0][0][e] = TPR
        results_1[1][0][e] = TNR
        results_1[2][0][e] = FPR
        results_1[3][0][e] = G_MEAN
        # TPR, TNR, FPR, G-MEAN for 5 second offset change point detection
        TN, FP, FN, TP = cf_matrix_5.ravel()
        TPR = TP/(TP+FN)
        TNR = TN/(TN+FP)
        FPR = FP/(FP+TN)
        G_MEAN = sqrt(TPR * TNR)
        results_5[0][0][e] = TPR
        results_5[1][0][e] = TNR
        results_5[2][0][e] = FPR
        results_5[3][0][e] = G_MEAN
        # TPR, TNR, FPR, G-MEAN for 10 second offset change point detection
        TN, FP, FN, TP = cf_matrix_10.ravel()
        TPR = TP/(TP+FN)
        TNR = TN/(TN+FP)
        FPR = FP/(FP+TN)
        G_MEAN = sqrt(TPR * TNR)
        results_10[0][0][e] = TPR
        results_10[1][0][e] = TNR
        results_10[2][0][e] = FPR
        results_10[3][0][e] = G_MEAN
        # detection delay
        detection_delay = get_detection_delay_from_predicted_label(y_pred, y.argmax(axis=-1), timestamps)
        detection_delays[0][e] = detection_delay
    # save population of results to file
    save_pop_results_to_file(RESULTS_DIR, results_1, 1, 0)
    save_pop_results_to_file(RESULTS_DIR, results_5, 5, 0)
    save_pop_results_to_file(RESULTS_DIR, results_10, 10, 0)
    np.savetxt(RESULTS_DIR + 'detection_delays/' + str(0) + '_detection_delay' + '.csv', detection_delays[0])
    # save trained models
    model_num = 0
    for model in models:
        model.save(RESULTS_DIR + 'word2vec_models/' + str(model_num) + '_execution.model')
        model_num += 1
    # mark experiment end
    print('... Experiment finished ...')
    print('Results saved to: ' + RESULTS_DIR)
    print('... ... ... ... ... ... ...')

if __name__ == "__main__":
    main(sys.argv)
