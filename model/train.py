import nltk
from nltk.stem import WordNetLemmatizer  # Note 1
import json  # data to be trainer
import pickle  # Performs Lin. Algebra
import numpy as np  # converting to np array
import tensorflow
from tensorflow.keras.models import Sequential  # Note 2
from tensorflow.keras.layers import Dense, Activation, Dropout  # Note 3
from tensorflow.keras.optimizers import SGD  # stocastic gradient desent
import random  # used shuffle

# check what these downloads are
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()  # set func var

words = []
classes = []
documents = []
ignore_words = ['?', '!']
intents = json.loads(open('model\intents.json').read())

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenizing words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add tokenized words to documents as an array
        documents.append((w, intent['tag']))
        # add classes to class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower())
         for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# DEBUG - DELETE LATER
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# pickle the words and classes - stores a file
pickle.dump(words, open('model\words.pkl', 'wb'))
pickle.dump(classes, open('model\classes.pkl', 'wb'))

# intialize training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []  # init bow
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)  # deprecated?
# create train and test limits. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print('Training data created')


# Creating model
# 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# ^ Why the specific layers
# equal to number of intents to predict output intent with softmax

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)
model.save('model\chatbot_model.h5', hist)


print('MODEL SAVED')
