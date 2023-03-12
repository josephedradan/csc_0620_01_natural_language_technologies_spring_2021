"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 2/15/2021

Purpose:
    Submission for hands-on workshop on Feb 11th

Details:

Description:
    Use an implementation of the Levenshtein distance algorithm in conjunction with the nltk library corpora of words
    to compare your given word to.

Notes:

IMPORTANT NOTES:

Explanation:

Reference:
    nltk words corpus does not contain “okay”?
        Note:
            Get all the words.
            manywords = words.words() + wordnet.words()  # Partially correct

        Reference:
            https://stackoverflow.com/questions/44449284/nltk-words-corpus-does-not-contain-okay


    2. Accessing Text Corpora and Lexical Resources
        Notes:
            Project Gutenberg electronic text archive, which contains some 25,000 free electronic books

        Reference:
            https://www.nltk.org/book/ch02.html

    python-Levenshtein 0.12.2
        Notes:
            The Documentation is bad so you gotta look up what the import is based on knowledge of python

        Reference:
            https://pypi.org/project/python-Levenshtein/
            https://rawgit.com/ztane/python-Levenshtein/master/docs/Levenshtein.html

    Is there a corpora of English words in nltk?
        Notes:
            Corpora for words

        Reference:
            https://stackoverflow.com/questions/28339622/is-there-a-corpora-of-english-words-in-nltk
"""
from collections import defaultdict
from typing import List, Dict, Set, Sequence, Tuple

from Levenshtein import StringMatcher
from nltk.corpus import gutenberg as GUTENBERG
from nltk.corpus import wordnet as WORDNET
from nltk.corpus import words as WORDS

"""
Get words from nltk Corpora and put them all into 1 list

:return:
"""
"""
Words come from the texts from the Gutenberg electronic text archive.
Has a bunch of crazy words.

"""
WORDS_GUTENBERG = GUTENBERG.words()

"""
Has a bunch of words.
Words are decent.
Contains unserializable (pickable) words, not good for threading

"""
WORDS_WORDS = WORDS.words()

"""
Has a bunch of words.
Contains words with underscores.
Has a bunch of crazy words.

"""
WORDS_WORDNET = [word for word in WORDNET.words()]

# Combined words from the 3 above corpora
WORDS_ALL = WORDS_WORDS + WORDS_WORDNET + WORDS_GUTENBERG


def get_dict_word_count(words: Sequence[str]) -> Dict[str, int]:
    """
    Get a dict with the Key as the word and the Value as the amount of times that word has been called

    :param words: list of words
    :return: dict with Key word and Value int
    """
    dict_word_frequency = defaultdict(int)

    for word in words:
        dict_word_frequency[word] += 1

    return dict_word_frequency


def get_dict_word_frequency(dict_word_count: Dict[str, int], word_count_total: int) -> Dict[str, float]:
    """
    Get a dict with the Key as the word and the Value as the percentage representing how many times that word has been
    called based on the total word count

    :param dict_word_count: dict with the Key as the word and Value as the amount of times that word has been said
    :param word_count_total: list containing all the words used to create dict_word_count
    :return: dict with Key word and Value float
    """
    dict_word_frequency = defaultdict(float)

    for word, count in dict_word_count.items():
        dict_word_frequency[word] = count / word_count_total

    return dict_word_frequency


class WordExistenceChecker:

    def __init__(self, words: Sequence[str]):
        self.words: Sequence[str] = words

        self.words_unique: Set[str] = set(self.words)

        self.amount_words: int = len(self.words)

        self.amount_words_unique: int = len(self.words_unique)

        self.dict_word_count: Dict[str, int] = get_dict_word_count(self.words)

        self.dict_word_frequency: Dict[str, float] = get_dict_word_frequency(self.dict_word_count,
                                                                             self.amount_words)

    def __call__(self, word: str) -> bool:
        """
        Check if the word in self.words_unique


        :param word: word
        :return: boolean
        """
        return word in self.words_unique

    def check_if_word_exists(self,
                             word: str,
                             amount: int = 5,
                             distance_sort_print: bool = True,
                             frequency_sort_print: bool = False,
                             double_sort_print: bool = False) -> None:
        """
        Check if the word in self.words_unique with printer that either prints if the word is in self.words_unique
        or print words that might be similar to the given word

        :param word: word
        :param amount: Amount of words similar to the given word
        :param distance_sort_print: Print based on Levenshtein distance
        :param frequency_sort_print: Print based on word frequency than Levenshtein distance
        :param double_sort_print: Print based on Levenshtein distance then word frequency
        :return: None
        """
        if self.__call__(word):
            print("{} is a complete and correct word in English based on self.words_unique.".format(word))
        else:
            print("{} is not a complete and correct word in English based on self.words_unique.\n".format(word))
            self.print_top_low_distance_words(word, amount, distance_sort_print, frequency_sort_print,
                                              double_sort_print)

    def get_levenshtein_min_distance_ztane(self, word: str) -> List[Tuple[str, int, float]]:
        """
        Loop through the unique words and create a list containing a tuple of data.
        The tuple of data should contain (Unique word, levenshtein minimum distance, frequency for the unique word)

        :param word: word
        :return: List of type tuple
        """

        # Create the list
        list_tuple = []

        # Loop through the set of unique words
        for word_unique in self.words_unique:
            # Temp tuple of the data
            tuple_temp = (word_unique,
                          StringMatcher.distance(word, word_unique),
                          self.dict_word_frequency.get(word_unique))

            # Append tuple to the list of type tuple
            list_tuple.append(tuple_temp)

        # Return the list of type tuple
        return list_tuple

    def print_top_low_distance_words(self, word, amount: int = 5,
                                     distance_sort_print=True,
                                     frequency_sort_print=False,
                                     double_sort_print=False):
        """
        Print the words similar to the given word if the word was not in self.words_unique

        :param word: word given
        :param amount: Amount of words similar to the given word
        :param distance_sort_print: Print based on Levenshtein distance
        :param frequency_sort_print: Print based on word frequency than Levenshtein distance
        :param double_sort_print: Print based on Levenshtein distance then word frequency
        :return:
        """
        list_tuple = self.get_levenshtein_min_distance_ztane(word)

        # Sort by Levenshtein distance
        list_tuple_sort_distance = sorted(list_tuple, key=lambda i: i[1])

        # Sort by word frequency
        list_tuple_sort_frequency = sorted(list_tuple, key=lambda i: i[2], reverse=True)

        # Sort by Levenshtein distance then word frequency
        list_tuple_sort_distance_frequency = []

        # Temp dict for "Sort by Levenshtein distance then word frequency"
        dict_list_tuple = defaultdict(list)

        # Double sorting for "Sort by Levenshtein distance then word frequency"
        for w, d, f in list_tuple:
            dict_list_tuple[d].append((w, d, f))

        for key in dict_list_tuple:
            dict_list_tuple[key].sort(key=lambda i: i[2], reverse=True)

        for key in sorted(dict_list_tuple.keys()):
            list_tuple_sort_distance_frequency.extend(dict_list_tuple.get(key))

        if distance_sort_print:
            print("Here are the {} closest words to the word {} by Levenshtein distance.".format(amount, word))
            for i in range(amount):
                print(self._get_str_tuple(*list_tuple_sort_distance[i]))
            print()

        if double_sort_print:
            print("Here are the {} closest words to the word {} by Levenshtein distance then word frequency.".format(
                amount, word))
            for i in range(amount):
                print(self._get_str_tuple(*list_tuple_sort_distance_frequency[i]))
            print()

        if frequency_sort_print:
            print("Here are the {} closest words to the word {} by word frequency.".format(amount, word))
            for i in range(amount):
                print(self._get_str_tuple(*list_tuple_sort_frequency[i]))
            print()

    @staticmethod
    def _get_str_tuple(*args):
        """
        Get a formatted string given the args which should be tuple of 3 args

        :param args: most likely a tuple of 3 items
        :return: string format of the 3 items
        """
        return "Word: {:<20} Distance: {:<3} Word Frequency: {}".format(args[0], args[1], args[2])


def demo():
    """
    Demo to test script for the workshop

    :return:
    """
    list_test_words = ["Joseph", "Edradan", "iaaa", "banana", "bananaa"]

    word_existence_checker_all = WordExistenceChecker(WORDS_ALL)
    word_existence_checker_words = WordExistenceChecker(WORDS_WORDS)
    word_existence_checker_wordnet = WordExistenceChecker(WORDS_WORDNET)
    word_existence_checker_gutenberg = WordExistenceChecker(WORDS_GUTENBERG)

    word_existence_checkers = [
        ("nltk.corpus.words", word_existence_checker_words),
        ("nltk.corpus.wordnet", word_existence_checker_wordnet),
        ("nltk.corpus.gutenberg", word_existence_checker_gutenberg),
        ("words, wordnet, gutenberg", word_existence_checker_all)]

    for corpora_used, word_existence_checker in word_existence_checkers:
        print("Using Corpus/Corpora {}".format(corpora_used))
        print("Amount of words: {}".format(word_existence_checker.amount_words))
        print("Amount of unique: {}".format(word_existence_checker.amount_words_unique))
        print()

    for word in list_test_words:
        print("#" * 50, "Word: {}".format(word), "#" * 50)
        print()
        for corpora_used, word_existence_checker in word_existence_checkers:
            print("Using Corpus/Corpora {}".format(corpora_used))
            word_existence_checker.check_if_word_exists(word,
                                                        distance_sort_print=False,
                                                        double_sort_print=True,
                                                        frequency_sort_print=False)
            print("\n" * 2, end="")


if __name__ == '__main__':
    demo()
