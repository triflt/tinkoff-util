import numpy as np
import argparse
from gensim.models import Word2Vec
from gensim.utils import tokenize
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--xgb_model', type=str, default ='data/xgb_model.json', help='Path to load the model')
    parser.add_argument('--w2v_model', type=str, default ='data/word2vec.model', help='Path to load word2vec')
    parser.add_argument('--dictionary', default='data/inv_dic.npy', type=str, help='Input to model length')
    parser.add_argument('--prefix', default=None, type=str, help='Input to model')
    parser.add_argument('--length', default=1, type=int,  help='Input to model length')

    arguments = parser.parse_args()


def sent2vec_generate(question: str, model, dim=32):
    token_question = np.array(list(tokenize(question, lowercase=True, deacc=True)))
    question_array = np.zeros(dim)
    count = 0
    for word in token_question:
        if model.wv.__contains__(str(word)):
            question_array += (np.array(model.wv[str(word)]))
            count += 1
    if count == 0:
        return question_array

    return question_array / count


def generate(ask, n):
    full_sentence = list(tokenize(ask, lowercase=True, deacc=True))
    for word in range(n):
        sent = [sent2vec_generate(ask, model_wv)]
        sorted_labels = np.sort(xgb_model_loaded.predict_proba(sent)[0])
        top_sorted_labels = sorted_labels[-10:-1]
        answer = inv_dict[int(xgb_model_loaded.predict(sent))]
        if answer in full_sentence:
            while answer in full_sentence:
                change = np.random.choice(top_sorted_labels)
                index_change = np.where(sorted_labels == change)[0][0]
                answer = inv_dict[index_change]
            ask = ask + ' ' + answer
            full_sentence.append(answer)
        else:
            ask = ask + ' ' + answer
            full_sentence.append(answer)
    ask += '...'
    print(ask)


model_wv = Word2Vec.load(arguments.w2v_model)
inv_dict = np.load(arguments.dictionary, allow_pickle=True).item()
xgb_model_loaded = pickle.load(open(arguments.xgb_model, "rb"))

generate(arguments.prefix, arguments.length)
