import json
import sys
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from scipy import spatial

# Kasteren dataset DIR
DIR = '../../kasteren_house_a/'
# Kasteren dataset file
DATASET_CSV = DIR + 'base_kasteren_reduced.csv'
# Word2Vec model
WORD2VEC_MODEL = DIR + 'actions_w1.model'
# Word2vec vector file
WORD2VEC_VECTOR_FILE = DIR + 'actions_w1_enhanced_graph.vector'
# If binary model is used or vector file is used
WORD2VEC_USE_FILE = True
# Embedding size
ACTION_EMBEDDING_LENGTH = 50
# Window size for custom context algorithm
WINDOW_SIZE = 2
# Minimum difference
REQUIRED_DIFFERENCE = 0.001

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

def create_action_embedding_matrix_from_file(tokenizer):
    data = pd.read_csv(WORD2VEC_VECTOR_FILE, sep=",", header=None)
    data.columns = ["action", "vector"]
    action_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(action_index) + 1, ACTION_EMBEDDING_LENGTH))
    unknown_words = {}    
    for action, i in list(action_index.items()):
        try:
            # print(data[data['action'] == action]['vector'].values[0])
            embedding_vector = np.fromstring(data[data['action'] == action]['vector'].values[0], dtype=float, sep=' ')
            # print(embedding_vector)
            embedding_matrix[i] = embedding_vector            
        except:
            if action in unknown_words:
                unknown_words[action] += 1
            else:
                unknown_words[action] = 1
    print(("Number of unknown tokens: " + str(len(unknown_words))))
    print (unknown_words)
    
    return embedding_matrix

def calculate_context_similarity(actions, embedding_action_matrix, position, window):
    target_action_vector = embedding_action_matrix[actions[position]]
    target_action_vector_context_sim = 0.0
    counter = window * 2
    for i in range(1, window+1):
        # right context we search for similarity
        if position+i < len(actions):
            right_sim = 1 - spatial.distance.cosine(target_action_vector, embedding_action_matrix[actions[position+i]])
            target_action_vector_context_sim += right_sim
        else:
            counter -= 1
        # left context we search for disimilarity (1 - similarity)
        if position-i >= 0:
            left_sim = 1 - max(0, 1 - spatial.distance.cosine(target_action_vector, embedding_action_matrix[actions[position-i]]))
            target_action_vector_context_sim += left_sim
        else:
            counter -= 1
    target_action_vector_context_sim = target_action_vector_context_sim / counter
    return target_action_vector_context_sim

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
    print("### ### ### ###")
    # remove None activity rows
    df_dataset = df_dataset[df_dataset['activity'] != "None"]
    # prepare dataset
    X, y, tokenizer_action = prepare_x_y_activity_change(df_dataset)
    # check prepared dataset struct
    print("### Actions ###")
    print(X)
    print("### ### ### ###")
    print("### Activity change ###")
    print(y)
    print("### ### ### ### ### ###")
    print("### Action Index ###")
    print(tokenizer_action.word_index)
    print("### ### ### ### ### ###")
    # create the  action embedding matrix for the embedding similarities calculation
    if WORD2VEC_USE_FILE:
        embedding_action_matrix = create_action_embedding_matrix_from_file(tokenizer_action)
    else:
        embedding_action_matrix, model, unknown_actions = create_action_embedding_matrix(tokenizer_action)
    # prepare required data structs
    counter = 0
    y_pred = []
    discovered_activities = []
    similarities = []
    discovered_activity_num = 0
    # check context similarity based on embeddings
    # doc: https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    for i in range(0, len(X)):
        context_similarity = calculate_context_similarity(X, embedding_action_matrix, i, WINDOW_SIZE)
        # print("Context similarity: " + str(context_similarity) + " Activity change: " + str(y[i]))
        similarities.append(context_similarity)
    # first action we assume no change of activity
    y_pred.append(0)
    discovered_activities.append('ACT' + str(discovered_activity_num))
    #  and detect activity change
    for i in range(0, len(X)-1):
        predicted_label = 1 if (similarities[i+1] - similarities[i] > REQUIRED_DIFFERENCE) else 0
        y_pred.append(predicted_label)
        if (predicted_label == 1):
            discovered_activity_num += 1
        discovered_activities.append('ACT' + str(discovered_activity_num))
    # print metrics
    print(classification_report(y, y_pred, target_names=['no', 'yes']))
    # save similarities and activity change labels to file
    similarity_act_change = {'action': df_dataset['action'], 'activity': df_dataset['activity'], 'similarity': similarities, 'activity_change_pred': y_pred, 'ground_truth': y}
    df_similarity_act_change = pd.DataFrame(data=similarity_act_change)
    df_similarity_act_change.to_csv('/results/similarities_and_activity_change_context_sim.csv', header=True, index=False, sep=',')
    # prepare dataset to be saved
    df_dataset['discovered_activity'] = discovered_activities
    df_dataset = df_dataset.drop("activity", axis=1)
    # save dataset with discovered activities (requires further processing)
    df_dataset.to_csv('/results/test_kasteren_context_sim.csv.annotated', header=None, index=True, sep=' ')

if __name__ == "__main__":
    main(sys.argv)
