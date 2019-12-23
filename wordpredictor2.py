import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import ipywidgets as widgets

SEQUENCELEN = 100

def buildmodel(VOCABULARY):
    model = Sequential()
    model.add(LSTM(256, input_shape = (SEQUENCELEN, 1), return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dense(VOCABULARY, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    return model
'''I trained this model on grail, but this can be changed to be another file if wanted
Originally I was going to train this on something like wikipedia articles, but each
epoch of training was taking several hours. Instead, I chose a smaller text like
grail that only takes about 10 minutes per epoch, and then did more epochs on it'''
file = open('data/grail.txt', encoding = 'utf8')
raw = file.read()
raw = raw.lower()

chars = sorted(list(set(raw)))
bad_chars = ['#', '*', '@', '_', '\ufeff']
for i in range(len(bad_chars)):
    raw = raw.replace(bad_chars[i],"")
chars = sorted(list(set(raw)))

textlen = len(raw)
charlen = len(chars)
VOCABULARY = charlen
print("Text length = " + str(textlen))
print("No. of characters = " + str(charlen))

char2int = dict((c, i) for i, c in enumerate(chars))
int2char = dict((c,i) for c,i in enumerate(chars))
inpstr = []
outstr = []

for i in range(len(raw) - SEQUENCELEN):
    X_text = raw[i: i + SEQUENCELEN]
    X = [char2int[char] for char in X_text]
    inpstr.append(X)
    Y = raw[i + SEQUENCELEN]
    outstr.append(char2int[Y])

length = len(inpstr)
inpstr = numpy.array(inpstr)
inpstr = numpy.reshape(inpstr, (inpstr.shape[0], inpstr.shape[1], 1))
inpstr = inpstr/float(VOCABULARY)

outstr = numpy.array(outstr)
outstr = np_utils.to_categorical(outstr)
print(inpstr.shape)
print(outstr.shape)
'''This is where you train the model. I have it set to save a model after each epoch, and then I hard-coded
which model to use based on how it saved to my computer. If using the model I uploaded, then this can be left alone,
otherwise will need to change this filename below to match after new training.'''
#model = buildmodel(VOCABULARY)
#filepath="saved_models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]

#history = model.fit(inpstr, outstr, epochs = 50, batch_size = 128, callbacks = callbacks_list)

filename = 'saved_models/weights-improvement-50-0.7394.hdf5'
model = buildmodel(VOCABULARY)
model.load_weights(filename)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

'''Get input of the appropriate length'''
usertext = input('Enter text up to 100 characters: ').lower()
if len(usertext) != 100:
    if len(usertext) > 100:
        usertext = usertext[-100:]
    else:
        while len(usertext) < 100:
            usertext += ' '
usertext = [char2int[c] for c in usertext]

NEWTXTLEN = int(input('How long of text do you want to generate (in integers)? '))
testtxt = usertext
newtxt = []

for i in range(NEWTXTLEN):
    X = numpy.reshape(testtxt, (1, SEQUENCELEN, 1))
    nextchar = model.predict(X/float(VOCABULARY))
    index = numpy.argmax(nextchar)
    newtxt.append(int2char[index])
    testtxt.append(index)
    testtxt = testtxt[1:]

print(''.join(newtxt))
