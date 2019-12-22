import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import pickle
import heapq
import os


path = '1661-0.txt'
text = open(path, encoding='utf8').read().lower()
tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)
uniqwords = np.unique(words)
uniqwordsindex = dict((c, i) for i, c in enumerate(uniqwords))
wlength = 5
prevwords = []
nextwords = []
for i in range(len(words) - wlength):
    prevwords.append(words[i:i + wlength])
    nextwords.append(words[i+ wlength])

X = np.zeros((len(prevwords), wlength, len(uniqwords)), dtype=bool)
Y = np.zeros((len(nextwords), len(uniqwords)), dtype=bool)
for i, each_words in enumerate(prevwords):
    for j, each_word in enumerate(each_words):
        X[i, j, uniqwordsindex[each_word]] = 1
    Y[i, uniqwordsindex[nextwords[i]]] = 1

if not os.path.exists('keras_next_word_model.h5'):
    model = Sequential()
    model.add(LSTM(128, input_shape=(wlength, len(uniqwords))))
    model.add(Dense(len(uniqwords)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history

    model.save('keras_next_word_model.h5')
    pickle.dump(history, open("history.p", "wb"))
else:
    model = load_model('keras_next_word_model.h5')
    history = pickle.load(open("history.p", "rb"))


def prepare_input(text):
    x = np.zeros((1, wlength, len(uniqwords)))
    for t, word in enumerate(text.split()):
        if word in uniqwords:
            x[0, t, uniqwordsindex[word]] = 1
    return x


def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predict_completions(text, n=3):
    if text == "":
        return("0")
    x = prepare_input(text)
    pred = model.predict(x, verbose=0)[0]
    next_indices = sample(pred, n)
    return [uniqwords[idx] for idx in next_indices]


q = "hello this is a test"
tokens = tokenizer.tokenize(q)
print("correct sentence: ",q)
seq = " ".join(tokenizer.tokenize(q.lower())[(len(tokens)-5):len(tokens)])
print("Sequence: ",seq)
print("next possible words: ", predict_completions(seq, 5))