import LSTM
import bigram
from nltk.tokenize import RegexpTokenizer


print('Please enter an incomplete sentence longer than 5 words: ')
text = input()
result = LSTM.inputString(text)

tokenizer = RegexpTokenizer(r'\w+')
lastword = list(tokenizer.tokenize(text))[-1]
bigramweb, bigramgod = bigram.getBigrams(lastword)
print(result)
print(bigramweb)
print(bigramgod)