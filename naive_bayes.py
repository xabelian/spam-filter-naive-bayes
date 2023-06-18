from math import log 
import csv
import nltk   
from nltk.tokenize import word_tokenize   
from collections import Counter 
import numpy as np
import pandas as pd

"""
This Naive Bayes Classifier is based on pseudocode described
in Speech and Language Processing, third edition by Daniel Jurafsky
"""

def split_train_test(data, train_percent=0.8):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    
    if not (0 <= train_percent <= 1):
        raise ValueError("Train percent must be a value between 0 and 1.")
    
    if data.empty:
        raise ValueError("Input data is empty.")
    # Randomize the data without specifying random_state
    data = data.sample(frac=1)  
    train_size = int(train_percent * len(data))
    
    if train_size == 0 or train_size == len(data):
        raise ValueError("Train percent is too small or too large.")
    
    # Reset index after randomizing
    train_df = data.iloc[:train_size].reset_index(drop=True)  
    test_df = data.iloc[train_size:].reset_index(drop=True)  
    
    return train_df, test_df


def get_document_text(df, column="message", class_name=None):

    """ Returns a string extracted from a column or class in a dataframe
    Parameters
        ----------
        df : Panda's Dataframe
            The dataframe we will extract data from
        column : str
            The column in the dataframe, by defect it's message since we
            are interested in extracting message the most.
        class_name : string
            The class name, can be either "spam" or "ham", used to extract
            all the text from a given class.
    """

    document_text = ""
    if class_name:
       document_text = df.loc[df['tag'] == class_name, column].str.cat(sep=' ')
    else:
       document_text = df[column].str.cat(sep=' ')

    return document_text

def get_vocabulary(document):

    """
    Given a document (that is a string), it returns a set with
    its vocabulary.
    """
    try:
        tokens = nltk.word_tokenize(document)
    except (LookupError, TypeError):
         raise ValueError("Error: Unable to tokenize the document.")

    vocab_set = set()

    for token in tokens:
        vocab_set.add(token)
    vocabulary = list(vocab_set)

    return vocabulary

def sum_word_counts(words, document, alpha=0.01):
    """

    """
    document = nltk.word_tokenize(document)
    wordcounts = Counter(document)
    total_count = 0
    for word in wordcounts:
          word_count = wordcounts[word]+alpha
          total_count += word_count
    return total_count

def word_ocurrs_by_document(document):
   document = nltk.word_tokenize(document)
   word_counts = Counter(document)
   return word_counts

def train_naive_bayes(df, classes, alpha=0.01, save_csv = False):
  """
  This should return a log P(c) and log P(w|c).
  We have to take the dataframe to count "documents", that is rows that are
  either spam or ham
  """
  logprior = {}
  bigdoc = {}
  loglikelihood = {}

  document = get_document_text(df)
  vocabulary = get_vocabulary(document)
  total_word_count_class = sum_word_counts(vocabulary, document)

  for c in classes:
    Ndoc = df.shape[0]
    Nc = (df['tag'] == c).sum()
    logprior[c]= log(Nc/Ndoc)
    bigdoc[c] = get_document_text(df, class_name=c)
    total_word_count_class = sum_word_counts(vocabulary, bigdoc[c])
    word_occur_class = word_ocurrs_by_document(bigdoc[c])
    for word in vocabulary:
      count_w_c = word_occur_class[word]
      loglikelihood[(word, c)] = log((count_w_c+alpha)/(total_word_count_class))
  
  return logprior, loglikelihood, vocabulary

def test_naive_bayes(testdoc, logprior, likelihood, classes, vocabulary):
  sum = {}
  testdoc = word_tokenize(testdoc)
  for c in classes:
    sum[c] = logprior[c]
    for word in testdoc:
      if word in vocabulary:
        sum[c] += likelihood[(word, c)]
  argmax_class = max(sum, key=sum.get)
  return argmax_class, sum[c]

def predict_class(testdoc, logprior, likelihood, classes, vocabulary):
    sum_values = {}
    testdoc = word_tokenize(testdoc)

    for c in classes:
        sum_values[c] = logprior[c]
        for word in testdoc:
            if word in vocabulary:
                sum_values[c] += likelihood[(word, c)]

    argmax_class = max(sum_values, key=sum_values.get)
    return argmax_class

def confusion_matrix(actual_classes, predicted_classes, labels):
    num_classes = len(labels)
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for actual, predicted in zip(actual_classes, predicted_classes):
        actual_index = labels.index(actual)
        predicted_index = labels.index(predicted)
        matrix[actual_index][predicted_index] += 1

    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)
    return confusion_df

## Metrics

def get_accuracy(confusion_df):

    tp = confusion_df.iloc[0,0]
    fp = confusion_df.iloc[0,1]
    fn = confusion_df.iloc[1,0]
    tn = confusion_df.iloc[1,1]
    accuracy = (tp+tn)/(tp+fp+tn+fn)

    return accuracy

def get_precision(confusion_df):
    
    tp = confusion_df.iloc[0,0]
    fp = confusion_df.iloc[0,1]
    fn = confusion_df.iloc[1,0]
    tn = confusion_df.iloc[1,1]
    precision = tp/(tp+fp)
    return precision