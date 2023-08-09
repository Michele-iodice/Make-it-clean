import numpy as np
import pandas as pd
import os
import re

import tf as tf
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    # generate a fixed number of characters
    for _ in range(n_chars):
        # encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict character
        yhat = model.predict_classes(encoded, verbose=0)
        # reverse map integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        # append to input
        in_text += char
    return in_text


def encode_seq(seq):
    sequences = list()
    for line in seq:
        # integer encode line
        encoded_seq = [mapping[char] for char in line]
        # store
        sequences.append(encoded_seq)
    return sequences


def create_seq(text):
    length = 30
    sequences = list()
    for i in range(length, len(text)):
        # select sequence of tokens
        seq = text[i - length:i + 1]
        # store
        sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))
    return sequences


def text_cleaner(text):
    # lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b", "", newString)
    # remove punctuations
    newString = re.sub("[^a-zA-Z]", " ", newString)
    long_words = []
    # remove short word
    for i in newString.split():
        if len(i) >= 3:
            long_words.append(i)
    return (" ".join(long_words)).strip()


def set_element(file, dir):
    x = ''
    f = open(dir + '/' + file, 'r')
    riga = f.readline()
    while riga != "":
        x += riga
        riga = f.readline()
    f.close()
    return x


def extract_data():
    data_text = ""
    dir = 'MLdataset'
    folder = os.listdir(dir)
    cont = 0
    for file in folder:
        cont = cont + 1
        x = set_element(file, dir)
        if len(x) > 0:
            data_text += x;

    print('CARICAMENTO DATASET TERMINATO ')
    # print(data_text)
    return data_text


if __name__ == '__main__':
    data = extract_data()
    # preprocess the text
    data_new = text_cleaner(data)
    # create sequences
    sequences = create_seq(data_new)
    # print(sequences)
    # create a character mapping index
    chars = sorted(list(set(data_new)))
    mapping = dict((c, i) for i, c in enumerate(chars))
    # encode the sequences
    sequences = encode_seq(sequences)
    # print(sequences)
    # vocabulary size
    vocab = len(mapping)
    sequences = np.array(sequences)
    # create X and y
    X, y = sequences[:, :-1], sequences[:, -1]
    # one hot encode y
    y = to_categorical(y, num_classes=vocab)
    # create train and validation sets
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    print('Train shape:', X_tr.shape, 'Val shape:', X_val.shape)

    # define model
    model = Sequential()
    model.add(Embedding(vocab, 50, input_length=30, trainable=True))
    model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(vocab, activation='softmax'))
    print(model.summary())

    # compile the model
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    # fit the model
    model.fit(X_tr, y_tr, epochs=100, verbose=2, validation_data=(X_val, y_val))
    # save the model
    model.save('Models/language_model_1')
    # load the model
    loaded_model = tf.keras.models.load_model('language_model_1')
    inp="large size of"
    print(len(inp))
    print(generate_seq(loaded_model, mapping, 30, inp.lower(), 15))
