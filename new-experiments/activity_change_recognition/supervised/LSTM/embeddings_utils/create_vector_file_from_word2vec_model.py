import sys
import argparse

from gensim.models import Word2Vec

# Results DIR
DIR = '/results/'

def main(argv):
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        type=str,
                        default='word2vec_context/',
                        nargs="?",
                        help="Dir for word2vec models")
    parser.add_argument("--model_folder",
                        type=str,
                        default='context_window_1_window_2_iterations_5_embedding_size_50/train/word2vec_models/',
                        nargs="?",
                        help="Folder for word2vec models")
    parser.add_argument("--model_name",
                        type=str,
                        default='0_execution.model',
                        nargs="?",
                        help="Name for word2vec model")
    parser.add_argument("--vector_file_name",
                        type=str,
                        default='0_execution.vector',
                        nargs="?",
                        help="Name for vector file name")
    args = parser.parse_args()
    # load word2vec model
    model = Word2Vec.load(DIR + args.model_dir + args.model_folder + args.model_name)
    # write model to vector file
    model.wv.save_word2vec_format(DIR + args.model_dir + args.model_folder + args.vector_file_name, binary=False)

if __name__ == "__main__":
    main(sys.argv)