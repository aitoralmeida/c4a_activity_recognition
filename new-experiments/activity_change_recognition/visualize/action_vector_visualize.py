import json
import sys

from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    # use tokenizer to generate indices for every action
    # very important to put lower=False, since the Word2Vec model
    # has the action names with some capital letters
    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(actions.tolist())
    # get model and extract embeddings
    model = Word2Vec.load(WORD2VEC_MODEL)    
    action_index = tokenizer.word_index
    # construct embeddings array
    embedding_array = []
    action_array = []
    unknown_actions = {}
    for action, i in list(action_index.items()):
        try:            
            embedding_vector = model[action]
            embedding_array.append(embedding_vector)
            action_array.append(action)        
        except:
            if action in unknown_actions:
                unknown_actions[action] += 1
            else:
                unknown_actions[action] = 1
    print(("Number of unknown tokens: " + str(len(unknown_actions))))
    print(unknown_actions)
    # fit PCA to reduce dim of embeddings
    pca = PCA(n_components=2)
    embedding_array_reduced_pca = pca.fit_transform(embedding_array)
    # fit TSNE to reduce dim of embeddings
    tsne = TSNE(n_components=2)
    embedding_array_reduced_tsne = tsne.fit_transform(embedding_array)
    # plot embeddings with action names (PCA)
    plot_embeddings_reduced(embedding_array_reduced_pca, action_array, '/results/actions_embedding_reduced_PCA_Window_5.png')
    # plot embeddings with action names (TSNE)
    plot_embeddings_reduced(embedding_array_reduced_tsne, action_array, '/results/actions_embedding_reduced_TSNE_Window_5.png')

if __name__ == "__main__":
    main(sys.argv)
