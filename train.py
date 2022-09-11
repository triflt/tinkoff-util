import numpy as np
import pickle
import argparse
import os
from xgboost import XGBClassifier
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim.utils import tokenize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--input_dir', type=str, help='Path to load the texts')
    parser.add_argument('--w2v_model', default='data/word2vec.model', type=str, help='Path to save w2v')
    parser.add_argument('--xgb_model', default='data/xgb_model.json', type=str, help='Path to save XGB')
    parser.add_argument('--dict', default='data/inv_dic.npy', type=str, help='Path to save dictionary')

    arguments = parser.parse_args()

    lines = []

    if arguments.input_dir:
        for file in os.listdir(arguments.input_dir):
            file = open(f"{arguments.input_dir}/{file}", "r", encoding="utf8")
            line = file.readlines()
            for lin in line:
                lines.append(lin)
    else:
        lines = str(input())


def tokenizer(txt):
    clean = []
    stop = ['с', 'но', 'к', 'и', 'что', 'где', 'a']
    for i in range(len(txt)):
        clean.append(list(tokenize(txt[i], lowercase=True, deacc=True)))
        clean[i] = [k for k in clean[i] if k not in stop]
    clean = [sentence for sentence in clean if sentence != []]
    return clean


def sent2vec_train(question: list, model, dim=32):
    token_question = np.array(question)
    question_array = np.zeros(dim)
    count = 0
    for word in token_question:
        if model.vocab().__contains__(str(word)):
            question_array += (np.array(model.vocab()[str(word)]))
            count += 1
    if count == 0:
        return question_array

    return question_array / count


def download(path):
    with open(path, 'r', encoding='utf8') as fin:
        text_1 = fin.readlines()
    return text_1


class XGBClassifierWrap:
    def __init__(self, n_estimators, max_depth, learning_rate):
        self.model = XGBClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   learning_rate=learning_rate,
                                   random_state=42,
                                   verbose=1)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def save(self, path):
        pickle.dump(self.model, open(path, "wb"))

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class Word2VecWrap:
    def __init__(self, tokens, vec_size, window, min_count, epochs):
        self.tokens = tokens
        self.epochs = epochs
        self.model_wv = Word2Vec(sentences=tokens,
                                 vector_size=vec_size,
                                 window=window,
                                 min_count=min_count)

    def train(self):
        self.model_wv.train(tokenized, total_examples=len(tokenized), epochs=self.epochs)

    def save(self, path):
        self.model_wv.save(path)

    def vocab(self):
        return self.model_wv.wv


def data_to_train(tokens):
    X = []
    Y = []
    for sentence in range(len(tokens)):
        X.append(sent2vec_train(tokens[sentence][:-1], model_wv, dim=32))
        Y.append(dic[tokens[sentence][-1]])
    return np.array(X), np.array(Y)


tokenized = tokenizer(lines)

dictionary = Dictionary(tokenized)
dic = dictionary.token2id
inv_dic = {v: k for k, v in dic.items()}
np.save(arguments.dict, inv_dic)

model_wv = Word2VecWrap(tokenized, vec_size=32, window=4, min_count=3, epochs=10)
model_wv.train()
model_wv.save(arguments.w2v_model)

X_train, Y_train = data_to_train(tokenized)

model_xgb = XGBClassifierWrap(n_estimators=5, max_depth=6, learning_rate=0.5)
model_xgb.fit(X_train, Y_train)
model_xgb.save(arguments.xgb_model)

print('ready!')
