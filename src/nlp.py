import ast
from string import punctuation

import nltk
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import (TreebankWordTokenizer)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

punctuation = list(punctuation)
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()


def tokenize(sentence):
    sentence = sentence.lower()
    return tokenizer.tokenize(sentence)


def normalize(words):
    return [lemmatizer.lemmatize(w) for w in words]


def stop_words(words):
    stop_word = set(stopwords.words('english'))
    return [w for w in words if w not in stop_word and w not in punctuation]


def vector_model(table):
    # create a list of lists of words from the 'table' DataFrame
    sentences = []
    for _, row in table.items():
        sentences.extend(ast.literal_eval(row))
    # sentences = [*(ast.literal_eval(row)) for _, row in table.items()]
    # train the Word2Vec model on all the sentences
    model = Word2Vec(sentences, min_count=1)
    return model


def word_vec(words, model, table):
    vectors = []
    for w in words:
        if w in model:
            if w not in table:
                table.append(w)
            vectors.append(model[w])
    sentence_vector = np.mean(vectors, axis=0)
    return sentence_vector
