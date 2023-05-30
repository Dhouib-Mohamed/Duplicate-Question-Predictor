from string import punctuation

import numpy as np
import spacy

lemmatizer = spacy.load("en_core_web_sm")

punctuation = list(punctuation)


def text_preprocessing(sentence, pbar, i):
    '''takes sentence as input and performs preprocessing steps including tokenization, lemmatization, lowercasing, and removal of stopwords and punctuation. It returns a list of preprocessed tokens ready for further analysis or modeling.\n
    pbar and i are variables to display the advancement of running the function'''
    if pbar is not None:
        pbar.update(i + 1)
    return [w.lemma_ for w in lemmatizer(sentence.lower()) if not w.is_stop and not w.is_punct]


def word_vec(words, model, table):
    '''calculates the sentence vector representation by taking a list of words, a pre-trained word embedding model, and a lookup table as inputs.'''
    vectors = []
    for w in words:
        if w in model:
            if w not in table:
                table.append(w)
            vectors.append(model[w])
    sentence_vector = np.mean(vectors, axis=0)
    return sentence_vector
