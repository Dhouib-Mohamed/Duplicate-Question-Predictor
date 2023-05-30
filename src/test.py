import src.nlp as nlp
import gensim.downloader as api
import importlib
import numpy as np

importlib.reload(nlp)


def test(model, word2vec, sentence1, sentence2):
    # Tokenization
    token1 = nlp.text_preprocessing(sentence1, None, 0)
    token2 = nlp.text_preprocessing(sentence2, None, 0)

    # Word Vectors
    token1_vec = nlp.word_vec(token1, word2vec, [])
    token2_vec = nlp.word_vec(token2, word2vec, [])

    token1_vec = [np.array(xi) for xi in token1_vec]
    token2_vec = [np.array(xi) for xi in token2_vec]

    token1_vec = np.array(token1_vec)
    token2_vec = np.array(token2_vec)

    # Reshape input tensors
    token1_vec = np.reshape(token1_vec, (1, 25))
    token2_vec = np.reshape(token2_vec, (1, 25))

    # Model Prediction
    res = model.predict([token1_vec, token2_vec])
    similarity_score = np.max(np.max(res))

    print("Similarity Score: ",similarity_score)
    print("Is duplicate: ",similarity_score>0.5)

