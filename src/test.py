import src.nlp as nlp
import gensim.downloader as api
import importlib
import numpy as np

importlib.reload(nlp)


def test(model,word2vec, sentence1, sentence2):
    # Tokenization
    token1 = nlp.tokenize(sentence1, 0, 0)
    token2 = nlp.tokenize(sentence2, 0, 0)

    # Word Vectors
    token1_vec = nlp.word_vec(token1, word2vec, [])
    token2_vec = nlp.word_vec(token2, word2vec, [])
    print([token1_vec, token2_vec])

    # Reshape input data
    token1_vec = np.expand_dims(token1_vec, axis=0)
    token2_vec = np.expand_dims(token2_vec, axis=0)

    # Model Prediction
    res = model.predict([token1_vec, token2_vec])
    print(res)
