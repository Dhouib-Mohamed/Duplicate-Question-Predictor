import src.nlp as nlp
import gensim.downloader as api
import importlib
import numpy as np

importlib.reload(nlp)


def test(model, word2vec, sentence1, sentence2):
    # Tokenization
    token1 = nlp.text_preprocessing(sentence1,None,0)
    token2 = nlp.text_preprocessing(sentence2,None,0)

    # Word Vectors
    token1_vec = nlp.word_vec(token1, word2vec,[])
    token2_vec = nlp.word_vec(token2, word2vec,[])

    token1_vec = [np.array(xi) for xi in token1_vec]
    token2_vec = [np.array(xi) for xi in token2_vec]

    token1_vec = np.array(token1_vec)  # questions 1 in the training set
    token2_vec = np.array(token2_vec)  # questions 2 in the training set

    # Reshape input tensors
    token1_vec = np.reshape(token1_vec, (1,) + token1_vec.shape)
    token2_vec = np.reshape(token2_vec, (1,) + token2_vec.shape)

    print([token1_vec, token2_vec])

    # Model Prediction
    res = model.predict([token1_vec, token2_vec])
    print(res)
