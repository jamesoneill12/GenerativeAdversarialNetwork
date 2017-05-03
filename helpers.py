# -*- coding: utf-8 -*-
from sklearn.cross_validation import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from collections import defaultdict
import os, json
from gensim.utils import simple_preprocess
tokenize = lambda x: simple_preprocess(x)
from keras.preprocessing.text import Tokenizer
flatten = lambda l: [item.strip() for sublist in l for item in sublist]

def splitter(sent):
    return str(sent).strip().lower().replace("\"", "").split(",")

def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def multilabel_classes(X,single=False):
    labeller = []
    lablist = [splitter(line) for line in X]
    uniq, ylabels = np.unique(flatten(lablist), return_inverse=True)
    labdict = dict([(item,i) for i, item in enumerate(uniq)])
    d = defaultdict(np.array)
    ohe = to_categorical(labdict.values(), nb_classes=len(labdict.keys()))
    for key, val in labdict.iteritems():
        d[key] = ohe[val]
    for labs in lablist:
        ohel = []
        for lab in labs:
            ohel.append(d[str(lab).strip()])
        ohel = np.array(ohel)
        labeller.append(np.sum(ohel,axis=0))
    if single:
        return uniq,ylabels,d, np.array(labeller)
    else:
        return d,np.array(labeller)

def create_embeddings(data_dir, embeddings_path, vocab_path, **params):
    class SentenceGenerator(object):
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):
            for sentence in self.dirname:
                yield tokenize(sentence)
    sentences = SentenceGenerator(data_dir)
    model = Word2Vec(sentences, **params)
    weights = model.syn0
    np.save(open(embeddings_path, 'wb'), weights)
    vocab = dict([(k, v.index) for k, v in model.vocab.items()])
    with open(vocab_path, 'wb+') as f:
        f.write(json.dumps(vocab))

def load_vocab(vocab_path='map.json'):
    """
    Load word -> index and index -> word mappings
    :param vocab_path: where the word-index map is saved
    :return: word2idx, idx2word
    """
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word

def word2vec_embedding_layer(embeddings_path):
    weights = np.load(open(embeddings_path, 'rb'))
    layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights])
    return layer

def get_embeddings(document,model,sent_len=100):
    max_words,max_length = model.syn0.shape[0],model.syn0.shape[1]
    arr = np.zeros(max_length, dtype='float32')
    B = []
    for sentence in document:
        A = []
        for word in sentence:
            try:
                emb =model[word]
                A.append(emb)
            except:
                A.append(arr)
        A = np.array(A)
        difference = sent_len - len(A)
        A = np.resize(A,(sent_len,max_length))
        B.append(A)
    return np.array(B)

def data_split(filepath, field='modality', multilabel=False, max_sequence=50, max_words=30000):

    params = {}
    root = "C:/Users/1/James/grctc/GRCTC_Project/Classification/"
    write_path = root+"GenerativeAdversarialNetwork/resources/embeddings/euro_embeddings.json"
    googleVecs = "C:/Users/1/James/grctc/GRCTC_Project/Classification/Data/" \
                 "Embeddings/word2vec/GoogleNews-vectors-negative300.bin"
    X = pd.read_csv(filepath, sep='\t', header='infer')
    model = Word2Vec.load_word2vec_format(googleVecs, binary=True)  # C binary format
    Xs = get_embeddings(X['sentence'],model)
    #weights = model.build_vocab(X['sentence'])
    #create_embeddings(X['sentence'], size=100, min_count=5, window=5, sg=1, iter=25)
    #word2idx, idx2word = load_vocab()

    if multilabel:
        uniq, ysingle, label_dict, y = multilabel_classes(X[field], single=True)
        multilabeldict = label_dict
    else:
        uniq, ysingle = np.unique(X[field], return_inverse=True)
        y = to_categorical(ysingle, nb_classes=len(uniq))
    indexes = range(len(Xs))
    data_train, data_test, labels_train, labels_test, \
    indices_train, indices_test = train_test_split(Xs, y, indexes,test_size=0.2,random_state=42)
    params = {'data_train': data_train, 'data_test': data_test, 'labels_train': labels_train,
                   'labels_test': labels_test, 'num_class': uniq, 'lab_train': ysingle[indices_train],
                   'lab_test': ysingle[indices_test], 'weights': Xs, 'field': field}
    return params