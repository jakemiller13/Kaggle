# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:52:10 2019

@author: jmiller
"""

import numpy as np
import pandas as pd
import os
import string
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, save_model, load_model
from keras.layers import Embedding, LSTM, Dense, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import datetime

# Load cleaned IMDB file to train sentiment analysis
cleaned_imdb = pd.read_csv('cleaned_imdb.csv', encoding = 'utf-8')

# Load Boston Airbnb datasets
def load_airbnb_datasets():
    '''
    Run this if you need to load in the Boston Airbnb datasets
    '''
    df_calendar = pd.read_csv('./Data/calendar.csv')
    df_listings = pd.read_csv('./Data/listings.csv')
    df_reviews = pd.read_csv('./Data/reviews.csv')
    return df_calendar, df_listings, df_reviews

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
                              
def load_GloVe(computer):
    '''
    Loads embedding matrix based on which computer you are using
    "computer": either "work" or "home"
    returns: embedding matrix
    '''
    GloVe_dict = dict()
    if computer == 'work':
        file_path = r'C:\Users\jmiller\Desktop\glove.6B.50d.txt'
    elif computer == 'home':
        file_path = './Data/glove.6B.50d.txt'
    with open(file_path, encoding = 'utf-8') as GloVe_file:
        for line in GloVe_file:
            values = line.split()
            word = values[0]
            coef = np.asarray(values[1:], dtype = 'float32')
            GloVe_dict[word] = coef
    return GloVe_dict

def strip_punctuation_and_whitespace(reviews_df, verbose = True):
    '''
    Strips all punctuation and whitespace from reviews EXCEPT spaces (i.e. ' ')
    Removes "<br />"
    Returns dataframe of cleaned imdb reviews
    '''
    trans_punc = str.maketrans(string.punctuation,
                               ' ' * len(string.punctuation))
    whitespace_except_space = string.whitespace.replace(' ', '')
    trans_white = str.maketrans(whitespace_except_space,
                                ' ' * len(whitespace_except_space))
    stripped_df = pd.DataFrame(columns = ['review', 'sentiment'])
    for i, row in enumerate(reviews_df.values):
        if i % 1000 == 0 and verbose == True:
            print('Stripping review: ' + str(i))
        if type(reviews_df) == pd.DataFrame:
            review = row[0]
            sentiment = row[1]
        elif type(reviews_df) == pd.Series:
            review = row
            sentiment = np.NaN
        try:
            review.replace('<br />', ' ')
            for trans in [trans_punc, trans_white]:
                review = ' '.join(str(review).translate(trans).split())
            combined_df = pd.DataFrame([[review, sentiment]],
                                       columns = ['review', 'sentiment'])
            stripped_df = pd.concat([stripped_df, combined_df],
                                    ignore_index = True)
        except AttributeError:
            continue
    return stripped_df

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

def create_tokenizer(max_words_to_keep, words_review_df):
    '''
    Creates tokenizer
    Function so it is simpler to adjust parameters
    '''
    tokenizer = Tokenizer(num_words = max_words_to_keep,
                          lower = True,
                          split = ' ')
    tokenizer.fit_on_texts(words_review_df['review'].values)
    return tokenizer, \
           tokenizer.texts_to_sequences(words_review_df['review'].values)

def pad_zeros(encoded_reviews, padding_length, padding = 'pre'):
    '''
    Pads integer reviews either left ('pre') or right ('post')
    '''
    return pad_sequences(encoded_reviews,
                         maxlen = padding_length,
                         padding = padding)

def coefficient_matrix(token, embedding):
    coef_matrix = np.zeros((token.num_words, len(embedding_dict['john'])))
    for word, i in token.word_index.items():
        if i < token.num_words:
            vector = embedding.get(word)
            if vector is not None:
                coef_matrix[i] = vector
    return coef_matrix

def create_flatten_model(vocab_length, in_length, weight_matrix,
                         opt = 'Adam', learning_rate = 0.001):
    '''
    Returns basic keras model
    vocab_length: vocabulary length
    in_length: imdb_sequence_length
    weight_matrix: coefficient matrix created from tokenizer
    opt = 'adam'
    learning rate = 0.001
    '''
    model = Sequential()
    model.add(Embedding(vocab_length,
                        weight_matrix.shape[1],
                        weights = [weight_matrix],
                        input_length = in_length,
                        trainable = False))
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))
    optimizer = getattr(keras.optimizers, opt)(lr = learning_rate)
    model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])
    return model

def create_LSTM_model(vocab_length, in_length, opt = 'Adam',
                      learning_rate = 0.001):
    '''
    Returns 1-layer LSTM model
    '''
    model = Sequential()
    model.add(Embedding(vocab_length, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation = 'sigmoid'))
    optimizer = getattr(keras.optimizers, opt)(lr = learning_rate)
    model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])
    return model

# Data has different folder structure depending on computer
location = input('Where are you ["work" or "home"]? \n')

# Create GloVe word embedding
embedding_dict = load_GloVe(location)

# Strip punctuation/white space from IMDB reviews
stripped_imdb = strip_punctuation_and_whitespace(cleaned_imdb)

# Plot histogram of review length - helps determine sequence length to use
imdb_lengths = get_length_all_reviews(stripped_imdb['review'])
plot_histogram(imdb_lengths, 1200)
imdb_sequence_length = 1000

# Tokenizer with 10000 word vocabulary. Pad zeros up to imdb_sequence_length
vocabulary_length = 10000
tokenizer, integer_reviews = create_tokenizer(vocabulary_length, stripped_imdb)
padded_reviews = pad_zeros(integer_reviews,
                           imdb_sequence_length,
                           padding = 'pre')

# Test/train split
split = 0.5
X_train, X_test, y_train, y_test = train_test_split(padded_reviews,
                                                    stripped_imdb['sentiment'],
                                                    test_size = split,
                                                    random_state = 42)

# Coefficient matrix for each word in training
coef_matrix = coefficient_matrix(tokenizer, embedding_dict)

# Create FLATTEN model - default is Adam optimizer with learning rate = 0.001
flatten_model = create_flatten_model(vocabulary_length,
                                     imdb_sequence_length,
                                     coef_matrix,
                                     opt = 'Adam',
                                     learning_rate = 0.001)
print(flatten_model.summary())
flat_history = flatten_model.fit(X_train, y_train,
                                 validation_data = (X_test, y_test),
                                 batch_size = 1000, epochs = 25, verbose = 1)

flatten_model.save('flatten_model_' + str(datetime.date.today()) + '.h5')

# CREATE LSTM model - default is Adam optimizer with learning rate = 0.001
LSTM_model = create_LSTM_model(vocabulary_length,
                               imdb_sequence_length,
                               opt = 'Adam',
                               learning_rate = 0.001)
print(LSTM_model.summary())
LSTM_history = LSTM_model.fit(X_train, y_train,
                              validation_data = (X_test, y_test),
                              batch_size = 1000, epochs = 10, verbose = 1)

LSTM_model.save('LSTM_model_' + str(datetime.date.today()) + '.h5')

# Load Airbnb data
df_calendar, df_listings, df_reviews = load_airbnb_datasets()

# Find indices which have >100 reviews, then find correlated listing ID
ids, counts = np.unique(df_reviews['listing_id'], return_counts = True)
gt_100 = np.where(counts > 100)[0]
ids_gt_100 = ids[gt_100]

ratings = {}

for temp_id in ids_gt_100:
    temp_comments = df_reviews.loc[df_reviews['listing_id'] == \
                                   temp_id]['comments']
    
    # Rename for function, then strip punctuation and whitespace
    temp_comments.rename('review', inplace = True)
    stripped_airbnb = strip_punctuation_and_whitespace(temp_comments,
                                                       verbose = False)
    
    # Plot histogram of review length
    airbnb_lengths = get_length_all_reviews(stripped_airbnb['review'])
    airbnb_sequence_length = 250
    
    # Tokenizer with 10000 word vocabulary
    airbnb_tokenizer, airbnb_integer_reviews = \
                                            create_tokenizer(vocabulary_length,
                                                             stripped_airbnb)
    # Pad zeros up to airbnb_sequence_length
    airbnb_padded_reviews = pad_zeros(airbnb_integer_reviews,
                                      airbnb_sequence_length,
                                      padding = 'pre')
    
    # Predict sentiment
    airbnb_sentiments = LSTM_model.predict_classes(airbnb_padded_reviews)
    predicted_rating = round(airbnb_sentiments.mean() * 100, 1)

    # Print comparisons
    actual_rating = df_listings.loc[df_listings['id'] == temp_id]\
                    ['review_scores_rating'].values[0]
    print('--- Listing ID ' + str(temp_id) + ' ---\nPredicted Rating: [' + \
          str(predicted_rating) + '] vs. Actual Rating: [' + \
          str(actual_rating) + ']')
    ratings[temp_id] = [actual_rating, predicted_rating]

# Sort ratings by ascending actual ratings
sorted_ratings = [ratings[i] for i in ratings]
sorted_ratings.sort()

# Separate actual and predicted ratings for plotting
plot_actual_ratings = [rating[0] for rating in sorted_ratings]
plot_predicted_ratings = [rating[1] for rating in sorted_ratings]

# Plot on separate axes for overlap
fig, ax1 = plt.subplots()
ax1.set_xlabel('Listing')
ax1.set_ylabel('LSTM Predicted Rating', color = 'orange')
predicted_line = ax1.plot(range(len(plot_predicted_ratings)),
                          plot_predicted_ratings,
                          color = 'orange',
                          label = 'Predicted Ratings')

ax2 = ax1.twinx()
ax2.set_ylabel('Actual Ratings', color = 'black')
actual_line = ax2.plot(range(len(plot_actual_ratings)),
                       plot_actual_ratings,
                       color = 'black',
                       label = 'Actual Ratings')
ax1.legend((predicted_line + actual_line),
           ['Predicted Rating', 'Actual Rating'],
           loc = 'upper center',
           bbox_to_anchor=(0.5, -0.15),
           fancybox = True,
           shadow = True,
           ncol = 2)
plt.title('Predicted Ratings vs. Actual Ratings for Boston Airbnbs')
plt.show()