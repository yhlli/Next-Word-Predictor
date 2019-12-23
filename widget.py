origtxt = []
predtxt = []

text = widgets.Text()
from IPython.display import display
display(text)

def handle_submit(sender):
    global predtxt
    global origtxt

    inp = list(text.value)

    lastword = inp[len(origtxt):]
    inp = inp[:len(origtxt)]
    origtxt = text.value
    lastword.append(' ')

    inptxt = [char_to_int[c] for c in inp]
    lastword = [char_to_int[c] for c in lastword]

    if len(inptxt) > 100:
        inptxt = inptxt[len(inptxt)-100: ]
    if len(inptxt) < 100:
        pad = []
        space = char_to_int[' ']
        pad = [space for i in range(100-len(inptxt))]
        inptxt = pad + inptxt

    while len(lastword)>0:
        X = np.reshape(inptxt, (1, SEQ_LENGTH, 1))
        nextchar = model.predict(X/float(VOCABULARY))
        inptxt.append(lastword[0])
        inptxt = inptxt[1:]
        lastword.pop(0)

    nextword = []
    nextchar = ':'
    while nextchar != ' ':
        X = np.reshape(inptxt, (1, SEQ_LENGTH, 1))
        nextchar = model.predict(X/float(VOCABULARY))
        index = np.argmax(nextchar)
        nextword.append(int_to_char[index])
        inptxt.append(index)
        inptxt = inptxt[1:]
        nextchar = int_to_char[index]

    predtxt = predtxt + [''.join(nextword)]
    print(" " + ''.join(nextword), end='|')

text.on_submit(handle_submit)

from tabulate import tabulate

origtxt1 = []
for item in origtxt:
    text_ = item.split()
    origtxt1.append(text_)
predtxt.insert(0,"")
predtxt.pop()

table = []
for i in range(len(origtxt1)):
    table.append([origtxt1[i], predtxt[i]])
print(tabulate(table, headers = ['Actual Word', 'Predicted Word']))
