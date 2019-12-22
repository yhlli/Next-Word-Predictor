from nltk import FreqDist
import nltk
nltk.download('webtext')
nltk.download('genesis')

STOPLIST = set(nltk.corpus.stopwords.words())


def is_content_word(word):
    return word.lower() not in STOPLIST and word[0].isalpha()


textweb = nltk.corpus.webtext.words()
textgod = nltk.corpus.genesis.words()
bigramsweb = [b for b in list(nltk.bigrams(textweb)) if is_content_word(b[1])]
bigramsgod = [b for b in list(nltk.bigrams(textgod)) if is_content_word(b[1])]
fd = nltk.ConditionalFreqDist(bigramsweb)
fd1 = nltk.ConditionalFreqDist(bigramsgod)


def getBigrams(text):
    genword = ''
    webword = ''
    if text in textgod:
        genword = list(fd1[text])[0]
    if text in textweb:
        webword = list(fd[text])[0]
    return webword, genword
