"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 4/6/2021

Purpose:

Details:

Description:

Notes:

IMPORTANT NOTES:

Explanation:

Reference:
    Load Pretrained glove vectors in python
        Notes:
        Reference:
            https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python

    numpy.argmin
        Notes:
            Get the location (indices) with the smallest argument

        Reference:
            https://numpy.org/doc/stable/reference/generated/numpy.argmin.html


    How do i find the iloc of a row in pandas dataframe?
        Notes:
            df.index.get_loc(window_stop_row.name)

        Reference:
            https://stackoverflow.com/questions/34897014/how-do-i-find-the-iloc-of-a-row-in-pandas-dataframe

    Check if a row exists in pandas
        Notes:
            'entry' in df.index

        Reference:
            https://stackoverflow.com/questions/45636382/check-if-a-row-exists-in-pandas

"""
import csv
import os
from functools import wraps
from typing import Union

import numpy as np
import pandas as pd
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

# 6 Billion tokens, 50d Vector
FILE_DATASET_6B_50D = os.path.join("H:/", "Datasets", "GloVe", "glove.6B.50d.txt")

# 840 Billion tokens, 300d Vector
FILE_DATASET_840B_300D = os.path.join("H:/", "Datasets", "GloVe", "glove.840B.300d.txt")

# 840 Billion tokens, 300d Vector word2vec version for gensim
FILE_DATASET_840B_300D_gensim_word2vec = os.path.join("H:/", "Datasets", "GloVe", "glove.840B.300d.txt")


def _get_pd_df_glove(text_file) -> pd.DataFrame:
    """

    Notes:
        Words will be row names


    :param text_file:
    :return:
    """
    pd_df = pd.read_table(text_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

    return pd_df


def get_pd_df_glove_840B_300D() -> pd.DataFrame:
    return _get_pd_df_glove(FILE_DATASET_840B_300D)


def get_pd_df_glove_6B_50D() -> pd.DataFrame:
    return _get_pd_df_glove(FILE_DATASET_6B_50D)


def get_vector_given_word(pd_df: pd.DataFrame, word_given) -> np.ndarray:
    # Must lower the word
    word_given = word_given.lower()
    return pd_df.loc[word_given].to_numpy()


def get_closest_word_given_vector_using_vector_sum(pd_df_given: pd.DataFrame, np_array_given: np.ndarray) -> str:
    """
    Given a vector of the dimensions of an array for a word in pd_df_6B,
    find the closest word to the vector via summing the entire row vector.

    Notes:
        works in a way that I did not intend it work in

    Reference:
        Load Pretrained glove vectors in python
            Reference:
                https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python

    :param pd_df_given:
    :param np_array_given:
    :return:
    """
    # Subtract np_array_given to all vectors
    np_array_diff = pd_df_given.to_numpy() - np_array_given

    # Sum the corresponding row for all rows into 1 column
    np_array_sum = np.sum(np_array_diff * np_array_diff, axis=1)

    # Get the index with the minimum value along axis 0 (rows)
    index = np.argmin(np_array_sum, axis=0)

    # Get word given the index found from np_argmin into the pd_df_given
    word_closet = pd_df_given.iloc[index].name

    return word_closet


def cosine_similarity(np_array_1: np.ndarray, np_array_2: np.ndarray) -> Union[int, float]:
    """


    Reference:
        Cosine Similarity between 2 Number Lists
            Reference:
                https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists

        Cosine similarity
            Notes:
                Cosine of 0 degrees = 1
                Cosine of (0, pi] degrees < 1
                0 > Cosine of (pi,180] degrees >= -1

                "
                two vectors with the same orientation have a cosine similarity of 1,
                two vectors oriented at 90° relative to each other have a similarity of 0, and two vectors
                diametrically opposed have a similarity of -1, independent of their magnitude.
                "
            Reference:
                https://www.wikiwand.com/en/Cosine_similarity


    :param np_array_1:
    :param np_array_2:
    :return:
    """

    # np.linalg.norm is l2 norm aka euclidean distance by default aka magnitude
    cosine_similarity = np.dot(np_array_1, np_array_2) / (np.linalg.norm(np_array_1) * np.linalg.norm(np_array_2))
    return cosine_similarity


def function_2_arg_wrapped(callable, np_array_2: np.ndarray):
    """
    Wrap a function's second argument

    :param np_array_2:
    :return:
    """

    @wraps(callable)
    def wrapper(np_array_1: np.ndarray):
        return callable(np_array_1, np_array_2)

    return wrapper


def get_closest_word_given_vector_using_cosine_similarity(pd_df_given: pd.DataFrame,
                                                          np_array_given: np.ndarray) -> str:
    """
    Using cosine similarity, find the closest word to the given vector

    Notes:
        Joseph had to figure this out,

        Cosine of 0 degrees = 1
        Cosine of (0, pi] degrees < 1
        0 > Cosine of (pi,180] degrees >= -1

    Reference:
        Apply function on each row (row-wise) of a NumPy array
            Reference:
                https://stackoverflow.com/questions/45604688/apply-function-on-each-row-row-wise-of-a-numpy-array

    :param pd_df_given:
    :param np_array_given:
    :return:
    """

    # Cosine similarity function with the second parameter loaded as np_array_given
    function_cosine_similarity = function_2_arg_wrapped(cosine_similarity, np_array_given)

    # Apply function_cosine_similarity over rows in pd_df as np_array
    np_array_cosine = np.apply_along_axis(function_cosine_similarity, 1, pd_df_given.to_numpy())
    # print("np_array_cosine")
    # print(np_array_cosine[0:50])

    # Get the index of np_array_cosine where the value is the closest to 0
    index = np.argmax(np_array_cosine, axis=0)

    # Get word given the index found from np_argmin into the pd_df_given
    word_closet = pd_df_given.iloc[index].name

    # print("INDEX:", index)
    # print("ARRAY:", np_array_cosine[index - 4:index + 5])
    # print("np_array_cosine.size:", np_array_cosine.size)

    return word_closet


def get_pd_df_words_given_vector_using_cosine_similarity(pd_df_given: pd.DataFrame,
                                                         np_array_given: np.ndarray) -> pd.DataFrame:
    """
    np array cosine similarity

    :param pd_df_given:
    :param np_array_given:
    :return:
    """
    name_cosine_similarity = "cosine_similarity"

    # Cosine similarity function with the second parameter loaded as np_array_given
    cosine_similarity_wrapped = function_2_arg_wrapped(cosine_similarity, np_array_given)

    # Apply cosine_similarity_wrapped over rows in pd_df as np_array
    np_array_cosine = np.apply_along_axis(cosine_similarity_wrapped, 1, pd_df_given.to_numpy())

    # Create new pd_df sorted by cosine_similarity
    pd_df_temp = pd.DataFrame(index=pd_df_given.index,
                              data={name_cosine_similarity: np_array_cosine}).sort_values(by=name_cosine_similarity,
                                                                                          ascending=False)
    return pd_df_temp


def distance_euclidean(np_array_1: np.ndarray, np_array_2: np.ndarray) -> pd.DataFrame:
    """
    Euclidean distance for 2 np arrays

    :param np_array_1:
    :param np_array_2:
    :return:
    """
    return np.power(np.sum(np.power((np_array_1 - np_array_2), 2)), .5)


def get_pd_df_words_given_vector_using_distance_euclidean(pd_df_given: pd.DataFrame,
                                                          np_array_given: np.ndarray) -> pd.DataFrame:
    """
    np array euclidean distance

    Reference:
        Pandas copy column names from one dataframe to another
            Notes:
                DataFrame(data=THE_DATA, columns=["NAME_1","NAME_2"])
                DataFrame(data={"NAME": A_DATA}

            Reference:
                https://stackoverflow.com/questions/56080859/pandas-copy-column-names-from-one-dataframe-to-another

        pandas.DataFrame.sort_values
            Notes:
                .sort_values(by="NAME_1")
                .sort_values(by=0)
                .sort_values(by=1)

            Reference:
                https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html

        How to append rows in a pandas dataframe in a for loop?
            Notes:
                In other words, do not form a new DataFrame for each row. Instead, collect all the data in a list of
                dicts, and then call df = pd.DataFrame(data) once at the end, outside the loop.

                Each call to df.append requires allocating space for a new DataFrame with one extra row, copying all
                the data from the original DataFrame into the new DataFrame, and then copying data into the new row.
                All that allocation and copying makes calling df.append in a loop very inefficient. The time cost of
                copying grows quadratically with the number of rows. Not only is the call-DataFrame-once code easier
                to write, it's performance will be much better -- the time cost of copying grows linearly with the
                number of rows.

            Reference:
                https://stackoverflow.com/questions/31674557/how-to-append-rows-in-a-pandas-dataframe-in-a-for-loop

    :param pd_df_given:
    :param np_array_given:
    :return:
    """

    name_distance_euclidean = "distance_euclidean"

    # Euclidean distance function with the second parameter loaded as np_array_given
    distance_euclidean_wrapped = function_2_arg_wrapped(distance_euclidean, np_array_given)

    # Apply distance_euclidean_wrapped over rows in pd_df as np_array
    np_array_euclidean = np.apply_along_axis(distance_euclidean_wrapped, 1, pd_df_given.to_numpy())

    # Create new pd_df sorted by cosine_similarity
    pd_df_temp = pd.DataFrame(index=pd_df_given.index,
                              data={name_distance_euclidean: np_array_euclidean}).sort_values(
        by=name_distance_euclidean,
        ascending=True)

    return pd_df_temp


def print_distance_euclidean_given_string(pd_df_given: pd.DataFrame, str_word, amount_similar_words=25):
    """
    Using gloves's python 3.4? code for finding cosine distance, it works most of the time if the word exists...

    Reference:
        GloVe/eval/python/distance.py
            Reference:
                https://github.com/stanfordnlp/GloVe/blob/master/eval/python/distance.py
            Notes:
                Code is in python 3.4? so I changed it to python3, though I don't know if i'm right...

        How do i find the iloc of a row in pandas dataframe?
            Reference:
                https://stackoverflow.com/questions/34897014/how-do-i-find-the-iloc-of-a-row-in-pandas-dataframe

        numpy.argsort
            Reference:
                https://numpy.org/doc/stable/reference/generated/numpy.argsort.html

    :param pd_df_given:
    :param str_word:
    :return:
    """
    W = pd_df_given.to_numpy()

    vec_result = None
    # print(str_word in pd_df_given.index)

    for idx, term in enumerate(str_word.split(' ')):
        if term in pd_df_given.index:
            # print('Word: %s  Position in vocabulary: %i' % (term, pd_df_given[term]))  # Old style
            print('Word: \"{}\" Position in vocabulary: {}'.format(term, pd_df_given.index.get_loc(term)))

            if idx == 0:
                # vec_result = np.copy(W[pd_df_given[term], :])  # Get the vector for the current word
                vec_result = get_vector_given_word(pd_df_given, term)
            else:
                # vec_result += W[pd_df_given[term], :]  # Old style
                vec_result += get_vector_given_word(pd_df_given, term)

        else:
            print('Word: %s  Out of dictionary!\n' % term)
            return

    # vec_norm = np.zeros(vec_result.shape)  # Unused

    # Euclidean distance formula (l2 norm)  # ?? Why transpose?
    distance = (np.sum(vec_result ** 2, ) ** (0.5))

    # vec_norm = (vec_result.T / distance).T  # ?? Why transpose?
    vec_norm = (vec_result / distance)

    # dist = np.dot(W, vec_norm.T)
    dist = np.dot(W, vec_norm)

    for term in str_word.split(' '):
        # index = pd_df_given[term]  # Old style
        index = pd_df_given.index.get_loc(term)
        dist[index] = -np.Inf

    np_array_sorted = np.argsort(-dist)[:amount_similar_words]

    print_format = "{:<35}{}"

    # print("\n                               Word       Cosine distance\n")  # Old style
    print(print_format.format("Word", "Cosine distance"))
    print("---------------------------------------------------------\n")
    for index in np_array_sorted:
        # print("%35s\t\t%f\n" % (pd_df_given.iloc[index], dist[index]))  # Old style
        print(print_format.format(pd_df_given.iloc[index].name, dist[index]))
    print()


def convert_glove_dataset_to_word2vec_format(file_glove_dataset_source: str, file_ginsim_word2vec_dataset_drain: str):
    """
    Convert glove dataset to word2vec dataset used for gensim

    Convert glove format to word2vec
        Notes:
        Reference:
            https://radimrehurek.com/gensim/scripts/glove2word2vec.html

    Gensim word vector visualization of various word vectors
        Notes:
            * Superior Stanford example compared to gensim exmaple

        Reference:
            https://web.stanford.edu/class/cs224n/materials/Gensim%20word%20vector%20visualization.html

    :param file_glove_dataset_source:
    :param file_ginsim_word2vec_dataset_drain:
    :return:
    """

    # Return if file word2vec version of file_glove_dataset_source already exists
    if os.path.exists(file_ginsim_word2vec_dataset_drain):
        print(f"word2vec of {file_glove_dataset_source} already exists!")

    # Create word2vec version of file_glove_dataset_source
    else:
        glove_file = datapath(file_glove_dataset_source)
        word2vec_glove_file = get_tmpfile(file_ginsim_word2vec_dataset_drain)
        glove2word2vec(glove_file, word2vec_glove_file)


if __name__ == '__main__':
    pd_df_6B = _get_pd_df_glove(FILE_DATASET_6B_50D)
    print("Priting PD DF Array")
    print(pd_df_6B.head())
    print()

    print(f"Testing {get_vector_given_word.__name__}")
    print(get_vector_given_word(pd_df_6B, "test"))
    print(type(get_vector_given_word(pd_df_6B, "love")))
    print()

    words_to_test = "obama"
    print(f"Testing cosine distance for the word: \"{words_to_test}\" ")
    print_distance_euclidean_given_string(pd_df_6B, words_to_test)
    print()

    word_to_test = "obama"
    print(f"Testing Joseph's custom cosine similarity function that he had to figure out: {word_to_test}")
    print(get_closest_word_given_vector_using_cosine_similarity(pd_df_6B,
                                                                get_vector_given_word(pd_df_6B, word_to_test)))
    print()

    words_to_test = "hello"
    print(f"Word to test: \"{words_to_test}\"")
    print(
        get_closest_word_given_vector_using_vector_sum(pd_df_6B, get_vector_given_word(pd_df_6B, words_to_test)))
    print()

    print(f"Testing {get_pd_df_words_given_vector_using_cosine_similarity.__name__} on {words_to_test}")
    print(get_pd_df_words_given_vector_using_cosine_similarity(pd_df_6B,
                                                               get_vector_given_word(pd_df_6B, words_to_test)))
    print()

    print(f"Testing {get_pd_df_words_given_vector_using_distance_euclidean.__name__} on {words_to_test}")
    print(get_pd_df_words_given_vector_using_distance_euclidean(pd_df_6B,
                                                                get_vector_given_word(pd_df_6B, words_to_test)))
    print()

    print("Classic: king - man + woman = queen using 840 Billion Tokens ()")
    pd_df_850B = get_pd_df_glove_840B_300D()

    vector_possible_queen = (get_vector_given_word(pd_df_850B, "king") -
                             get_vector_given_word(pd_df_850B, "man") +
                             get_vector_given_word(pd_df_850B, "woman")
                             )
    print(get_pd_df_words_given_vector_using_cosine_similarity(pd_df_850B, vector_possible_queen))  # 1m 35s
    print(get_pd_df_words_given_vector_using_distance_euclidean(pd_df_850B, vector_possible_queen))  # 38s
    print()
