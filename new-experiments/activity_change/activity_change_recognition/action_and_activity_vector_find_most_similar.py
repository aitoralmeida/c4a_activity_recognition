import json
import sys

from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from scipy.spatial import distance

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
# Number of elements in the action's embbeding vector
ACTION_EMBEDDING_LENGTH = 50

def plot_embeddings_reduced(embeddings, actions, filename):
    fig, ax = plt.subplots()
    ax.scatter(embeddings[:, 0], embeddings[:, 1])
    for i, txt in enumerate(actions):
        ax.annotate(actions[i], (embeddings[:, 0][i], embeddings[:, 1][i]))
    ax.axis('equal')
    fig.savefig(filename)

def main(argv):
    print(('*' * 20))
    print('Loading dataset...')
    sys.stdout.flush()
    # dataset of activities
    DATASET = DATASET_CSV
    df_dataset = pd.read_csv(DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]    
    # recover all the actions in order
    actions = df_dataset['action'].values
    print(('total actions', len(actions)))
    # recover all the activity in order
    activities = df_dataset['activity'].values
    # use tokenizer_actions to generate indices for every action
    # very important to put lower=False, since the Word2Vec model
    # has the action names with some capital letters
    tokenizer_actions = Tokenizer(lower=False)
    tokenizer_actions.fit_on_texts(actions.tolist())
    # use tokenizer_activities to generate indices for every activity
    tokenizer_activities = Tokenizer(lower=False)
    tokenizer_activities.fit_on_texts(activities.tolist())
    # get model and extract embeddings of actions
    model = Word2Vec.load(WORD2VEC_MODEL)    
    action_index = tokenizer_actions.word_index
    # construct action embeddings array
    embedding_action_array = []
    action_array = []
    unknown_actions = {}
    for action, i in list(action_index.items()):
        try:            
            embedding_vector = model[action]
            embedding_action_array.append(embedding_vector)
            action_array.append(action)        
        except:
            if action in unknown_actions:
                unknown_actions[action] += 1
            else:
                unknown_actions[action] = 1
    print(("Number of unknown tokens: " + str(len(unknown_actions))))
    print(unknown_actions)
    # construct activity embeddings dict (action avg)
    activity_index = tokenizer_activities.word_index
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
    # get dict values into two separated arrays
    embedding_activity_array = list(embedding_activity_action_dict.values())
    activity_array = list(embedding_activity_action_dict.keys())
    # find most similar with cosine similarity measure
    for i in range(0, len(embedding_action_array)):
        distances = distance.cdist([embedding_action_array[i]], embedding_activity_array, "cosine")[0]
        min_index = np.argmin(distances)
        min_distance = distances[min_index]
        max_similarity = 1 - min_distance
        print("Action " + action_array[i] + " matches: " + activity_array[min_index])

if __name__ == "__main__":
    main(sys.argv)
