# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:52:10 2019

@author: jmiller
"""

'''
This script was written prior to some serious exploration in Keras
As such, there are a lot of functions that are much more easily handled
using keras modules/functions
'''

# https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow
# https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/

import numpy as np
import pandas as pd
import csv
import os
import string
import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt

# Load cleaned IMDB file to train sentiment analysis
cleaned_imdb = pd.read_csv('cleaned_imdb.csv', encoding = 'utf-8')

# Load Boston Airbnb datasets
df_calendar = pd.read_csv('./Data/calendar.csv')
df_listings = pd.read_csv('./Data/listings.csv')
df_reviews = pd.read_csv('./Data/reviews.csv')

def clean_imdb(directory):
    '''
    Returns cleaned dataframe of IMDB reviews
    home computer directory: './Data/IMDB/'
    Saved as 'cleaned_imdb.csv'
    '''
    sentiment = {'neg': 0, 'pos': 1}
    df_columns = ['review', 'sentiment']
    reviews_with_sentiment = pd.DataFrame(columns = df_columns)
    for i in ('test', 'train'):
        for j in ('neg', 'pos'):
            file_path = directory + i + '/' + j
            for file in os.listdir(file_path):
                with open((file_path + '/' + file), 'r',
                          encoding = 'utf-8') as text_file:
                    text = text_file.read()
                review = pd.DataFrame([[text, sentiment[j]]],
                                      columns = df_columns)
                reviews_with_sentiment = reviews_with_sentiment.\
                                         append(review, ignore_index = True)
    return reviews_with_sentiment
                              
def load_embedding_matrix(computer):
    '''
    Loads embedding matrix based on which computer you are using
    "computer": either "work" or "home"
    returns: embedding matrix
    '''
    if computer == 'work':
        file_path = r'C:\Users\jmiller\Desktop\glove.6B.50d.txt'
    elif computer == 'home':
        file_path = './Data/glove.6B.50d.txt'
    word_matrix = pd.read_table(file_path,
                                sep = ' ',
                                index_col = 0,
                                header = None,
                                quoting = csv.QUOTE_NONE)
    return word_matrix

def create_word_list(word_embedding):
    '''
    Slices off first columns from "word_matrix"
    Returns a list of words
    '''
    return word_embedding.index.tolist()

def strip_punctuation_and_whitespace(reviews_df):
    '''
    Strips all punctuation and whitespace from reviews EXCEPT spaces (i.e. ' ')
    Removes "<br />"
    Returns dataframe of cleaned imdb reviews
    '''
    stripped_df = pd.DataFrame(columns = reviews_df.columns)
    trans_punc = str.maketrans(string.punctuation,
                               ' ' * len(string.punctuation))
    whitespace_except_space = string.whitespace.replace(' ', '')
    trans_white = str.maketrans(whitespace_except_space,
                                ' ' * len(whitespace_except_space))
    for i, row in enumerate(reviews_df.values):
        if i % 1000 == 0:
            print('Stripping sentence: ' + str(i))
        review = row[0]
        sentiment = row[1]
        review.replace('<br />', ' ')
        for trans in [trans_punc, trans_white]:
            review = ' '.join(str(review).translate(trans).split())
        combined_df = pd.DataFrame([[review, sentiment]],
                                   columns = ['review', 'sentiment'])
        stripped_df = pd.concat([stripped_df, combined_df],
                                ignore_index = True)
    return stripped_df

def get_max_sentence_length(sentences):
    '''
    Returns length of longest sentence (this is NOT index)
    '''
    split_sentences = [sentence.split(' ') for sentence in sentences]
    index_of_longest = split_sentences.index(max(split_sentences, key = len))
    return len(sentences[index_of_longest])

def get_length_all_reviews(sentences):
    '''
    Returns a list of length of all reviews
    Used for plotting histogram
    '''
    lengths = [len(i.split(' ')) for i in sentences]
    return lengths

def plot_histogram(sentence_lengths, x_dim):
    '''
    Plots histogram of length of all sentences
    '''
    plt.hist(sentence_lengths, 50, [0, x_dim])
    plt.xlabel('Review length (words)')
    plt.ylabel('Frequency')
    plt.title('Review Lengths (Words per review)')
    plt.show()

def words_to_integers(words_reviews_df, list_of_words, max_sequence_length):
    '''
    Turns "list_of_sentences" into sequence of integers, one sentence per line
    These may be able to be turned into arrays for optimization
    '''
    integer_df = pd.DataFrame(columns = words_reviews_df.columns)
    for i, row in enumerate(words_reviews_df.values):
        if i % 1000 == 0:
            print('Translating words to integers in review: ' + str(i))
        review = row[0].split(' ')[-max_sequence_length:]
        sentiment = row[1]
        word = 0
        while word < max_sequence_length:
            try:
                review[-word - 1] = list_of_words.\
                                    index(review[-word - 1].lower())
            except ValueError:
                review[-word - 1] = 0
            except IndexError:
                review.insert(0, 0)
            finally:
                word += 1
        combined_df = pd.DataFrame([[review, sentiment]],
                                   columns = ['review', 'sentiment'])
        integer_df = pd.concat([integer_df, combined_df], ignore_index = True)
    return integer_df

def equal_length_matrix(integer_sentences, sequence_length):
    '''
    Currently not being used
    Creates a matrix of zeros that is:
        [len(integer_sentences) x sequence_length]
    Then fills integers from right-hand side
    '''
    matrix = np.zeros((len(integer_sentences), sequence_length), dtype = int)
    for i, sentence in enumerate(integer_sentences):
        sentence = np.array(sentence)
        matrix[i, -len(sentence):] = sentence[-sequence_length:]
    return matrix    

def split_data(review_df, percentage):
    '''
    Percentage is how much data will be in train set
    Returns X_train, y_train, X_test, y_test
    Note that df.loc is inclusive of both start/stop
    '''
    split = int(percentage * review_df.shape[0])
    X_train = review_df.loc[:split, :]['review']
    y_train = review_df.loc[:split, 'sentiment'].values
    X_test = review_df.loc[split:, :]['review']
    y_test = review_df.loc[split:, 'sentiment'].values
    return X_train, y_train, X_test, y_test, split

def create_model(data_dim, timesteps, out_dim, opt = 'Adam',
                 learning_rate = 0.001):
    '''
    Returns keras model
    # out_dim = 32 or 128
    # input_shape = (imdb_sequence_length, split)
    # opt = 'adam'
    # learning rate = 0.01
    '''
    model = Sequential()
    model.add(Embedding(5000, 32, input_length = imdb_sequence_length))
    model.add(LSTM(128, return_sequences = True))
    model.add(LSTM(64, return_sequences = False))
    model.add(Dense(1, activation = 'softmax'))
    optimizer = getattr(keras.optimizers, opt)(lr = learning_rate)
    model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])
#    model = Sequential()
#    model.add(LSTM(128, return_sequences = True,
#                   input_shape = (timesteps, 1)))
#    model.add(LSTM(64, return_sequences = False))
#    model.add(Dense(1, activation = 'softmax'))
#    optimizer = getattr(keras.optimizers, opt)(lr = learning_rate)
#    model.compile(loss = 'binary_crossentropy',
#                  optimizer = optimizer,
#                  metrics = ['accuracy'])
    return model

# Data has different folder structure depending on computer
location = input('Where are you ["work" or "home"]? \n')

# Create word embedding and word list
word_vectors = load_embedding_matrix(location)
word_list = create_word_list(word_vectors)

# Strip punctuation/white space from IMDB reviews
stripped_imdb = strip_punctuation_and_whitespace(cleaned_imdb)

# Plot histogram of review length - helps determine sequence length to use
imdb_lengths = get_length_all_reviews(stripped_imdb['review'])
plot_histogram(imdb_lengths, 1200)
imdb_sequence_length = 1000

# List of integer sentences, one sentence per line. Takes a LONG time
int_list = words_to_integers(stripped_imdb, word_list, imdb_sequence_length)

# Creates matrix of integer sentences with 0-padding on left side
# integer_sentence_matrix = equal_length_matrix(int_list, imdb_sequence_length)

# Create train/test sets
X_train, y_train, X_test, y_test, split = split_data(int_list, 0.5)

# Create model - default is Adam optimizer with learning rate = 0.001
model = create_model(data_dim = split,
                     timesteps = imdb_sequence_length,
                     out_dim = 32,
                     opt = 'Adam',
                     learning_rate = 0.001)
print(model.summary())

# Train model
Xtrain = X_train.values.reshape(-1, X_train.shape[0])
Xtest = X_test.values.reshape(-1, X_test.shape[0])

history = model.fit(Xtrain, y_train, validation_data = (Xtest, y_test),
                    batch_size = 100, epochs = 5, verbose = 1)