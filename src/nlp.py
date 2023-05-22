from string import punctuation

import numpy as np
import spacy

lemmatizer = spacy.load("en_core_web_sm")

punctuation = list(punctuation)


def tokenize(sentence, pbar, i):
    pbar.update(i + 1)
    return [w.lemma_ for w in lemmatizer(sentence.lower()) if not w.is_stop and not w.is_punct]


def word_vec(words, model, table):
    vectors = []
    for w in words:
        if w in model:
            if w not in table:
                table.append(w)
            vectors.append(model[w])
    sentence_vector = np.mean(vectors, axis=0)
    return sentence_vector
