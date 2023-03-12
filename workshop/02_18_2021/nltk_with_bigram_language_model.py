"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 2/22/2021

Purpose:
    Given a word, find the bigram that predicts the next word and that bigram's probability

Details:

Description:

Notes:
    1) Build the bigram LM:
        1.A) Use nltk to compile all the unique bigrams from the corpus you used for the previous assignment.
        1.B) Compute probability of each bigram using MLE ( count(w1 w2) / count(w1) )

    2) Next word prediction using the above bigram LM:
        2.A) Get an input word from user, inpW.
        2.B) Use the above bigram LM to find all the bigrams where the input word, inpW, is w1.
        Display all possible next words from these bigrams and their corresponding probabilities.

IMPORTANT NOTES:

Explanation:

Reference:

"""
import traceback
from collections import defaultdict
from typing import List, Tuple, Any, Dict

import nltk
from nltk.corpus import gutenberg
from nltk.corpus import gutenberg as GUTENBERG
from nltk.corpus import stopwords

# All words from gutenberg.words()
WORDS_GUTENBERG = GUTENBERG.words()

# Not used
stopwords_eng: List[str] = stopwords.words('english')


def generate_gutenberg_info() -> Tuple:
    """
    From the gutenberg module:
        load the words from the corpora
        Load all the texts:
            get bigrams from those sentences  # bigram == pair

    Warning:
        The sentences from the gutenberg.sents function does not remove stopping words and
        sequence of chars such as ",", "!--", ";", etc...

    TODO:
        Remove sequence of chars and stopping words

    :return: dict pair and its count, dict word and its count, all sentences from the corpora
    """
    # List of sentences
    sentences = []

    # Dict key: word, value: count
    dict_word_count = defaultdict(int)

    for word in WORDS_GUTENBERG:
        dict_word_count[word] += 1

    # Dict key: pair, value: count
    dict_pair_count = defaultdict(int)  # Alternatively you can use lambda: 1 to start at 1 than 0

    # Loop through all gutenberg texts and create bigrams from each sentence
    for filename in gutenberg.fileids():
        sentences: List[List[str]] = gutenberg.sents(filename)

        for sentence in sentences:
            bigram = nltk.bigrams(sentence)

            for pair in bigram:
                # print(i)
                dict_pair_count[pair] += 1

    return dict_pair_count, dict_word_count, sentences


def generate_pair_info(dict_pair_count: Dict[Tuple[str], int], dict_word_count: Dict[str, int]) -> Tuple:
    """
    Create the dict pair and its probability with a dict word with all its pairs

    :param dict_pair_count: dict key pair, value count based on the corpora
    :param dict_word_count: dict key word, value word based on the corpora
    :return: dict_word_pair, dict_pair_probability
    """
    dict_word_pair = defaultdict(list)

    dict_pair_probability = defaultdict(float)

    # Loop through each pair and find its bigram probability while making a dict with word and pair
    for pair, count_pair in dict_pair_count.items():
        word_beginning = pair[0]

        dict_word_pair[word_beginning].append(pair)

        count_word_temp = dict_word_count.get(word_beginning)

        # If the beginning word does not exist in the dict_word_pair
        if count_word_temp is None:
            print(f"{word_beginning} is not a word in dict_word_count.")
            continue

        try:
            dict_pair_probability[pair] = count_pair / count_word_temp

        except Exception as e:
            print(pair, count_pair, count_word_temp)
            print(traceback.print_exc())

    return dict_word_pair, dict_pair_probability


def generate_bigram_info_gutenberg() -> Tuple:
    """
    Generate the bigrams and their corresponding probability based on the gutenberg corpora

    :return: dict_word_pair (word and starting word in the bigram),
             dict_pair_probability (bigram and probabiilty),
             dict_pair_count (bigram and count based on corpora),
             dict_word_count (word and count based on corpora),
             sentences (all sentences in teh corpora)
    """
    print("Gutenberg corpora was loaded")
    dict_pair_count, dict_word_count, sentences = generate_gutenberg_info()

    print("Gutenberg bigram probabilities generated")
    dict_word_pair, dict_pair_probability = generate_pair_info(dict_pair_count, dict_word_count)

    return dict_word_pair, dict_pair_probability, dict_pair_count, dict_word_count, sentences


class BigramWordPredictor:
    """
    An object that is easy for the user to find the bigram information about the word they give to this
    and also all the bigrams from the corpora

    """

    def __init__(self, bigram_info: Tuple[Any]):
        """
        From the given bigram information, unpack it and associate it with an instance variable

        :param bigram_info: Information about the bigrams
        """
        self.dict_word_pair = bigram_info[0]
        self.dict_pair_probability = bigram_info[1]
        self.dict_pair_count = bigram_info[2]
        self.dict_word_count = bigram_info[3]
        self.sentences = bigram_info[4]

        # Not used
        self.list_v_pair_v_prob = [(pair, prob) for pair, prob in self.dict_pair_probability.items()]

        # Not used
        self.list_v_pair_v_prob_sorted = sorted(self.list_v_pair_v_prob,
                                                key=lambda i: (i[0][0], i[1]),
                                                reverse=True)

    def print_bigram_probability(self) -> None:
        """
        Print all the bigrams from the info created with this object

        :return: None
        """
        for pair, probability in self.dict_pair_probability.items():
            print(f"{pair}: {probability}")

    def predict_next_word_probability_from_bigram(self, word: str, amount_spacing: int = 30) -> None:
        """
        Given the word. find the bigrams and their associated probability that predicts the next following word
        (It may not necessarily be a word)

        Note:
            a word from the bigram may not be a word, it might be a sequence of chars that represent something.

        :param word:
        :param amount_spacing:
        :return:
        """
        pairs = self.dict_word_pair.get(word)

        if pairs is None:
            print(f"Word: {word} does not exist in the corpora!")
            print()
            return

        list_v_pair_v_prob = [(pair, self.dict_pair_probability.get(pair)) for pair in pairs]

        list_info_bigram: List[Tuple[Tuple[str], float]] = sorted(list_v_pair_v_prob,
                                                                  key=lambda i: (i[0][0], i[1]),
                                                                  reverse=True)

        print(f"For the word {word} "
              f"here are the possible following words/sequence of chars in a pair with their corresponding probability:")
        for pair, prob in list_info_bigram:
            # print("Pair: {0:<{1}} Probability: {2}".format(str(pair),
            #                                                amount_spacing,
            #                                                prob))
            print("Pair: {0:<{1}} {2:<{3}} {4}".format(str(pair),
                                                       amount_spacing,
                                                       "P( {0} | {1} ):".format(pair[1],
                                                                                pair[0]),
                                                       amount_spacing,

                                                       prob))
        print()


if __name__ == '__main__':
    info = generate_bigram_info_gutenberg()

    bigram_word_predictor = BigramWordPredictor(info)
    print()

    bigram_word_predictor.predict_next_word_probability_from_bigram("Joseph")
    bigram_word_predictor.predict_next_word_probability_from_bigram("Edradan")
    bigram_word_predictor.predict_next_word_probability_from_bigram("Smith")
