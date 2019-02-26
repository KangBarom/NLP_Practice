from random import randint
import pandas as pd
import matplotlib.pyplot as plt
import csv
import codecs
import numpy as np
import torch
# Load model
from models import InferSent
import sys
import nltk
# nltk.download('punkt')
#
# model_version = 2
# MODEL_PATH = "/home1/InferSent/encoder/infersent%s.pickle" % model_version
# params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
#                 'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
# model = InferSent(params_model)
# model.load_state_dict(torch.load(MODEL_PATH))
# # Keep it on CPU or put it on GPU
# use_cuda = True
# model = model.cuda() if use_cuda else model
# W2V_PATH = '/home1/InferSent/oov_train_model.vec'
# model.set_w2v_path(W2V_PATH)
# # Load embeddings of K most frequent words
# # model.build_vocab_k_words(K=100000)
# model.build_vocab_k_words(K=2051129) # word embedding된 모든 단어 추출
#
# # Load test sentences
#
# train_test = pd.read_csv('/home1/InferSent/testset.csv',header = None, delimiter=",",  encoding='UTF-8')
# source_s = train_test[0][1:]
# target_s = train_test[1][1:]

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

#print(cosine(model.encode(['the cat eats.'])[0], model.encode(['the cat drinks.'])[0]))


# embeddings_source = model.encode(source_s, bsize=128, tokenize=False, verbose=True)
# print('nb source_s encoded : {0}'.format(len(embeddings_source)))
# embeddings_target = model.encode(target_s, bsize=128, tokenize=False, verbose=True)
# print('nb target_s encoded : {0}'.format(len(embeddings_target)))
# np.save('embeddings_source.npy', embeddings_source)
# np.save('embeddings_target.npy', embeddings_target)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='InferSent unsupervied learning using evidnet dataset.')
    parser.add_argument('--model_version', required=False,
                        default="2",
                        metavar="/path/to/infersent.pickle",
                        help="Path to weight for GloVe (1) or Fasttext (2)")
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download setting and make numpy vectors',
                        type=bool)
    parser.add_argument('--cosine', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='make cosine vectors and download vector in local directory',
                        type=bool)
    args = parser.parse_args()
    print("download: ", args.download)
    print("Model: ", args.model_version)
    print("Makeing cosine vector : ", args.cosine)

    if args.download == True:
        nltk.download('punkt')
        model_version = args.model_version
        MODEL_PATH = "/home1/InferSent/encoder/infersent%s.pickle" % model_version
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
        model = InferSent(params_model)
        model.load_state_dict(torch.load(MODEL_PATH))
        # Keep it on CPU or put it on GPU
        use_cuda = True
        model = model.cuda() if use_cuda else model
        W2V_PATH = '/home1/InferSent/oov_train_model.vec'
        model.set_w2v_path(W2V_PATH)
        # Load embeddings of K most frequent words
        # model.build_vocab_k_words(K=100000)
        model.build_vocab_k_words(K=2051129)  # Extract embedding word .

        # Load test sentences

        train_test = pd.read_csv('/home1/InferSent/testset.csv', header=None, delimiter=",", encoding='UTF-8')
        source_s = train_test[0][1:]
        target_s = train_test[1][1:]
        embeddings_source = model.encode(source_s, bsize=128, tokenize=False, verbose=True)
        print('nb source_s encoded : {0}'.format(len(embeddings_source)))
        embeddings_target = model.encode(target_s, bsize=128, tokenize=False, verbose=True)
        print('nb target_s encoded : {0}'.format(len(embeddings_target)))
        np.save('embeddings_source.npy', embeddings_source)
        np.save('embeddings_target.npy', embeddings_target)

    if args.cosine == True:
        source_np = np.load('embeddings_source.npy')
        target_np = np.load('embeddings_target.npy')
        print('Success vector load')
        # Load for checking the vector name.
        train_test = pd.read_csv('/home1/InferSent/testset.csv', header=None, delimiter=",", encoding='UTF-8')
        # ground-truth dataset
        source_s = train_test[0][1:]
        target_s = train_test[1][1:]

        # Shuffle test data sets for accuracy verification.
        # np.random.shuffle(target_np)
        score_arr_5 = []  #  Store idx in the target per source.

        top_N = 5
        y =np.ones(len(target_s))
        p=[]
        print('source_np size: ', len(source_np))
        print('target_np size: ', len(target_np))
        print('source_s size: ', len(source_s))
        for i in range(1, len(source_np)+1):
            score_perSource = []
            for j in range(1, len(target_np)+1):
                temp_val = cosine(source_np[i-1], target_np[j-1])
                score_perSource.append(temp_val)
            topN_arr=np.argsort(score_perSource)[::-1][:top_N]
            if np.any(topN_arr == i):
                p.append(1)
            else:
                p.append(0)
            score_arr_5.append(topN_arr)  # top 5 idx
            with open("result/unsupervised_top5_result.txt", 'a') as f:
                f.write("===============%d=================\n" % (i))
                f.write("Source : %s \n" % (str(source_s[i])))
                f.write("Truth : %s \n" % (str(target_s[i])))
                f.write("Predict : \n")
                for k in topN_arr:
                    f.write("   %s \n" % (str(target_s[k+1])))

            printProgress(i, len(source_np), 'Progress:', 'Complete', 1, 50)

        # Assuming that the correct answer is in the top N, calculate f1 score.
        import sklearn.metrics as metrics

        print('accuracy', metrics.accuracy_score(y, p))
        print('precision', metrics.precision_score(y, p))
        print('recall', metrics.recall_score(y, p))
        print('f1', metrics.f1_score(y, p))
        print(metrics.classification_report(y, p))
        print(metrics.confusion_matrix(y, p))
        with open("result/unsupervised_top5_result_matrix.txt", 'a') as f:
            f.write('accuracy', metrics.accuracy_score(y, p))
            f.write('precision', metrics.precision_score(y, p))
            f.write('recall', metrics.recall_score(y, p))
            f.write('f1', metrics.f1_score(y, p))
            f.write(metrics.classification_report(y, p))
            f.write(metrics.confusion_matrix(y, p))

        np.save('vector_result/unsuper_source2target_top5.npy', score_arr_5)










