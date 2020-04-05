import nltk
# Natural Language Toolkit
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# This stemmer object ginds the root word of a word
# Eg
# program, programmer, programs, programming will all result in the same word program which is
# is the root of all the above words.

import numpy
# Numpy will be used for array modifications.
import tflearn
# tflearn will be used for making the Deep Learning Neural network.
import random
# To choose random responses from the data saved in the intents.json file.
import json
# To read the data from the intents.json file.
import pickle
# To save details of the model, the training data and the output so we wouldn't need to compute
# it again and again.

with open("intents.json") as f:
    data = json.load(f)


# print(data["intents"])    #This prints the data in intents file, execute this if you're confused

# If you're running it for the first time, there will be no pickle file in your working directory
# which will result in an error when you try and open it. The except block will thus be executed.
try:
    raise Exception("Uncomment to run the except block")
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # We thus need to tokenize the pattern. This just basically means that we need to
            # separate out the phrases into words.
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    wordstemp = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(wordstemp)))
    labels=sorted(labels)

    training=[]
    output=[]

    # Since neural networks only understand numbers and not strings, we need to encode the input
    # data ie. We take each tokenized phrase and encode it into 1s and 0s
    for index, doc in enumerate(docs_x):
        encoding=[]
        # This encoded list will be the input to the neural network.
        wrds = [stemmer.stem(w.lower()) for w in doc  if w != "?"]

        for w in words:
            if w in wrds:
                encoding.append(1)
            else:
                encoding.append(0)
        output_row = [0]*len(labels)
        output_row[labels.index(docs_y[index])] = 1
        # The output to the neural network is a list of 0s and 1s such that a 1 at index i
        # represents that the neural network thinks that the phrase is of label at ith index.
        training.append(encoding)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)
    # Converted list to a numpy array.
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
    # Saved the data as objects into a pickle file.
# Now we're gonna use the ML models available to us.

net = tflearn.input_data(shape=[None, len(training[0])])    # The input layer of the DNN
net = tflearn.fully_connected(net, 8)                       # First hidden layer
net = tflearn.fully_connected(net, 8)                       # Second hidden layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")   # Output layer
net = tflearn.regression(net)
# The above is explaining the system how to optimize the output of the DNN. The default
# is the adam optimizer.

model = tflearn.DNN(net)

try:
    raise Exception("Uncomment to run the except block")
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=500, batch_size = 8, show_metric=True)
    model.save("model.tflearn")

# This function converts the input phrase that the user entered into a DNN readable encoded format
def createInput(userString):
    global words
    
    phrase = nltk.word_tokenize(userString)
    phrase = [stemmer.stem(word.lower()) for word in phrase]

    newEncoding = []
    for single_word in words:
        if single_word in phrase:
            newEncoding.append(1)
        else:
            newEncoding.append(0)
    # Both the above method and the following method will work seamlessly. Do take a look.
    # newEncoding = [0]*len(words)
    # for i, single_word in enumerate(phrase):
    #     if single_word in phrase:
    #         newEncoding[i] = 1
    #     else:
    #         newEncoding[i] = 0

    return numpy.array(newEncoding)

def chat():
    print("Start talking with the bot!!\n(Write quit to quit)")
    while True:
        response = input("You: ")
        if response.lower() == "quit":
            break
        result = model.predict([createInput(response)])
        result_index = numpy.argmax(result)
        # print(result[0][result_index])
        # It returns an array of results. But, since in our case we only have one result, we can use 0 indexing to 
        # fetch the first result.
        FinalTag = labels[result_index]
        # FinalResponse = random.choice(data["intents"][tag]["patterns"]) is not possible since data["intents"]
        # return a list and not a dictionary
        if result[0][result_index]>0.7:
            for elements in data["intents"]:
                if FinalTag == elements["tag"]:
                    FinalResponse = random.choice(elements["responses"])
                    break
            print(FinalResponse)
        else:
            print("I didn't understand. Would you care to repeat?")
        
chat()