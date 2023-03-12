"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 4/6/2021

Purpose:

Details:

Description:

Notes:
    Test word2vec using gensim

IMPORTANT NOTES:

Explanation:

Reference:
    Word2Vec with Gensim - Python
        Reference:
            https://www.youtube.com/watch?v=Z1VsHYcNXDI

    models.word2vec – Word2vec embeddings
        Reference:
            https://radimrehurek.com/gensim/models/word2vec.html#


"""
from typing import Generator

import nltk
from gensim.models import Word2Vec

from handler_squad_2 import get_gen_text_train


# print(json.loads(FILE_DATASET_TRAIN))
# with open(FILE_DATASET_TRAIN, encoding="UTF-8") as f:
#     print(json.load(f))
# pd_df_sentences = pd.read_json(FILE_DATASET_TRAIN)
# print(pd_df_sentences[0:-1])

def get_list_tokenize_string_nltk(iterable_given) -> Generator:
    for words in iterable_given:
        yield nltk.word_tokenize(words)


def _test_word2vec():
    list_token = list(get_list_tokenize_string_nltk(get_gen_text_train()))
    model_word2vec = Word2Vec(list_token, min_count=1, size=32)
    print(model_word2vec.wv.most_similar("test"))


if __name__ == '__main__':
    _test_word2vec()
