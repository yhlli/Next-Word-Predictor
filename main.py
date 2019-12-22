import LSTM
import bigram
from nltk.tokenize import RegexpTokenizer
import trigrams


print('Please enter an incomplete sentence at least longer than 3 words (preferably 5) TYPE \'exit\' to exit: ')
text = input()
while text != 'exit':
    result = LSTM.inputString(text)

    tokenizer = RegexpTokenizer(r'\w+')
    lastword = list(tokenizer.tokenize(text))[-1]
    bigramweb, bigramgod = bigram.getBigrams(lastword)
    trigram = trigrams.getTrigrams(list(tokenizer.tokenize(text))[-2], lastword)

    if bigramweb != '':
        result.append(bigramweb)
    if bigramgod != '':
        result.append(bigramgod)
    if trigram != '':
        result.append(trigram)
    print('Enter the number of the correct word, 0 for custom word:')
    i = 1
    for word in result:
        print(str(i)+': '+ str(result[i-1]))
        i+=1
    number = input()
    if str(number)=='exit':
        break;
    while int(number)>i-1 and int(number)!=0:
        print('Please enter a valid number')
        number = input()
    if int(number)!= 0:
        text = text + ' ' +  result[int(number)-1]
    if int(number)== 0:
        print('Sorry, please type the correct word:')
        correct = input()
        text = text + ' ' + correct
    print('SENTENCE: ' + text)
print('Thank you for using our next word predictor!')