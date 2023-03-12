"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 3/8/2021

Purpose:
    Uses text classification using Naive Bayes

Details:

Description:
    1) Create a toy labeled dataset. (Use any data structure in python to store this data).
    This dataset should have 10-15 sentences with a class label associated with each datapoint.
    There should be at least 2 classes (e.g. spam vs ham OR positive vs negative)

    2) Calculate the prior probabilities.

    3) Calculate the conditional probabilities of each word (with respect to both the categories) in the dataset.

    4) Take a sentence as input from the user.

    5) Using the probabilities calculated before and Naive Bayes classification, predict the class of the input sentence.
    (Calculate the probability of the input sentence being in different classes.
    The class having the highest probability will be the output class.)

Notes:
    Data:
        Covid News Text:
            https://www.cnn.com/2021/03/08/health/covid-19-vaccine-cdc-guidelines-fully-vaccinated/index.html
            https://www.washingtonpost.com/nation/2021/03/09/coronavirus-covid-live-updates-us/
            https://www.nytimes.com/live/2021/03/08/world/covid-19-coronavirus
            https://www.latimes.com/california/story/2021-03-08/la-county-expecting-largest-covid-19-vaccine-shipment
            https://ktla.com/news/california/still-far-from-herd-immunity-california-could-face-a-4th-covid-19-wave/

        Non Covid News Text:
            https://www.bbc.com/news/world-us-canada-56323906
            https://www.bbc.com/news/uk-56329887
            https://www.bbc.com/news/uk-england-northamptonshire-56326406
            https://www.cnn.com/2021/03/09/us/les-miles-kansas-football-trnd-spt/index.html
            https://www.cnn.com/2021/03/09/asia/first-ever-quad-leaders-summit-intl-hnk/index.html
            https://www.cnn.com/2021/03/09/uk/royal-family-harry-meghan-succession-intl/index.html
            https://www.cnbc.com/2021/03/09/linkedin-gender-a-major-work-opportunity-barrier-for-women-in-asia-.html

    The model should technically be less accurate since i'm doing Covid news and Non Covid news.
    You are better off looking at the Laplace smoothing answers.
    Notice that the Non Covid News is about the Royal family because that was the top story at the time of making this
    algorithm.

IMPORTANT NOTES:

Explanation:

Reference:
    38. argmax and argmin in Numpy
        Reference:
            https://www.youtube.com/watch?v=zyQVmq2rq_I

    Naive Bayes, Clearly Explained!!!
        Reference:
            https://www.youtube.com/watch?v=O2L2Uv9pdDA

    Machine Learning | Laplace Correction
        Reference:
            https://www.youtube.com/watch?v=mY3765RW6Uw

    Gaussian Naive Bayes, Clearly Explained!!!
        Reference:
            https://www.youtube.com/watch?v=H3EjCKtlVog

    Tokenize a paragraph into sentence and then into words in NLTK
        Reference:
            https://stackoverflow.com/questions/37605710/tokenize-a-paragraph-into-sentence-and-then-into-words-in-nltk
"""
import math
import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Union

import nltk


def filter_text_to_words_v1(text: str) -> List[str]:
    """
    1. Get text then
    2. Return list of words

    Reference:
        How to remove punctuation in python?
            https://stackoverflow.com/questions/53664775/how-to-remove-punctuation-in-python

    :param text:
    :return:
    """

    # Remove all non alphanumeric except apostrophe
    text = re.sub('[^A-Za-z0-9\']+', ' ', text)

    # Split by any whitespace
    words = text.split()

    return words


def filter_text_to_words_v2(text: str) -> List[str]:
    """
    1. Get text then
    2. Return list of words

    :param text:
    :return:
    """
    # Capture all chars or chars.chars or chars'chars or chars`chars
    words = re.findall(r"[\w]+[.'`-][\w]+|\w+", text)
    return words


def get_dict_word_count(text: str) -> Tuple[Dict[Any, int], int]:
    """
    1. Get text
    2. Convert text to list of words
    3. Return dict key word, value count

    :param text:
    :return:
    """
    word_count = 0

    words = filter_text_to_words_v2(text)

    dict_word_count = defaultdict(int)

    for word in words:
        word_count += 1

        dict_word_count[word] += 1

    return dict_word_count, word_count


def get_dict_word_frequency(dict_given: Dict[str, int], word_count) -> Dict[str, float]:
    """
    Convert dict key word, value count to dict key word, value frequency

    :param dict_given:
    :param word_count:
    :return:
    """
    dict_word_probability = defaultdict(float)

    for word, count in dict_given.items():
        dict_word_probability[word] = count / word_count

    return dict_word_probability


def get_text_from_file(filename: str) -> str:
    """
    Read text from a given filename (the text is in the cwd so filename is fine)

    :param filename:
    :return:
    """
    with open(filename, encoding="utf-8") as file:
        text = file.read()

    return text


def get_sentences_from_text(text) -> List[str]:
    """
    Use nltk sentence tokenizer to get sentences

    :param text:
    :return:
    """
    list_sentence = nltk.sent_tokenize(text)

    return list_sentence


class TextClassificationContainer:
    """
    IDK name
    """

    def __init__(self, classification: str):
        self.classification = classification

        self.amount_word: Union[int, None] = None
        self.text: Union[str, None] = None
        self.dict_k_word_v_count: Union[Dict[str, int], None] = None
        self.dict_k_word_v_frequency: Union[Dict[str, float], None] = None

        self.sentences: Union[List[str], None] = None
        self.amount_sentence: Union[int, None] = None

        self.probability_prior: Union[float, None] = None
        self.dict_k_word_v_count_laplace_smoothing: Union[Dict[str, int], None] = {}
        self.dict_k_word_v_frequency_laplace_smoothing: Union[Dict[str, float], None] = {}

    def __str__(self):
        return "{:<25}{}\n{:<25}{}\n{:<25}{}\n{:<25}{}".format(
            "Classification:",
            self.classification,
            "Amount of words:",
            self.amount_word,
            "Amount of sentences:",
            self.amount_sentence,
            "Prior Probability:",
            self.probability_prior)


class NaiveBayes:
    """
    Naive Bayes
    """

    def __init__(self, dict_k_classification_v_file_abs_path: dict, alpha=1):
        """
        Naive bayes algorithm as a class

        :param dict_k_classification_v_file_abs_path: Classification and the text file associated with it
        :param alpha: alpha used for Laplace smoothing
        """
        self.dict_k_classification_v_file_abs_path: dict = dict_k_classification_v_file_abs_path
        self.alpha = alpha

        self.dict_k_classification_v_text_classification_container: Union[Dict[str,
                                                                               TextClassificationContainer],
                                                                          None] = {}

        for classification, file_abs_path in self.dict_k_classification_v_file_abs_path.items():
            self.dict_k_classification_v_text_classification_container[classification] = TextClassificationContainer(
                classification)

        self.amount_sentence_total: int = 0
        self.words_total: Union[List[str], None] = []

        self._load_pre()

        self.amount_word_unique = 0

        self._load_post()

    def _load_pre(self):
        """
        Pre calculation, creats the text_classification_container objects

        :return: None
        """
        for classification, file_abs_path in self.dict_k_classification_v_file_abs_path.items():
            text = get_text_from_file(file_abs_path)

            self.dict_k_classification_v_text_classification_container[classification].text = text

            dict_k_word_v_count, amount_word = get_dict_word_count(text)

            self.words_total.extend(dict_k_word_v_count)

            dict_k_word_v_frequency = get_dict_word_frequency(dict_k_word_v_count, amount_word)

            # Set up amount_word
            self.dict_k_classification_v_text_classification_container[
                classification].amount_word = amount_word

            # Set up dict_k_word_v_count
            self.dict_k_classification_v_text_classification_container[
                classification].dict_k_word_v_count = dict_k_word_v_count

            # Set up dict_k_word_v_frequency
            self.dict_k_classification_v_text_classification_container[
                classification].dict_k_word_v_frequency = dict_k_word_v_frequency

            sentences = get_sentences_from_text(text)
            amount_sentence = len(sentences)

            self.amount_sentence_total += amount_sentence

            # Set up sentences
            self.dict_k_classification_v_text_classification_container[
                classification].sentences = sentences

            # Set up amount_sentence
            self.dict_k_classification_v_text_classification_container[
                classification].amount_sentence = amount_sentence

    def _load_post(self):
        """
        Post calculation after the creation of text_classification_container objects

        :return: None
        """
        # Calculate amount_word_unique
        self.amount_word_unique = len(set(self.words_total))

        for classification, text_classification_container in self.dict_k_classification_v_text_classification_container.items():

            # Set up probability_prior
            text_classification_container.probability_prior = text_classification_container.amount_sentence / self.amount_sentence_total

            # Set up dict_k_word_v_count_laplace_smoothing
            for word in self.words_total:
                if word in text_classification_container.dict_k_word_v_count:
                    text_classification_container.dict_k_word_v_count_laplace_smoothing[word] = \
                        text_classification_container.dict_k_word_v_count[word] + 1
                    continue
                text_classification_container.dict_k_word_v_count_laplace_smoothing[word] = 1

            # Set up dict_k_word_v_frequency_laplace_smoothing
            text_classification_container.dict_k_word_v_frequency_laplace_smoothing = get_dict_word_frequency(
                text_classification_container.dict_k_word_v_count_laplace_smoothing,
                text_classification_container.amount_word + self.amount_word_unique
            )

    def print_text_classification_container(self):
        """
        Print information about the text_classification_container objects
        :return:
        """
        for classification, text_classification_container in self.dict_k_classification_v_text_classification_container.items():
            print(text_classification_container)
            print()

    def _classify_text_helper(self, list_word: List[str],
                              text_classification_container: TextClassificationContainer,
                              laplace_smoothing: bool = False,
                              ) -> float:
        """
        Multiply probabilities

        Notes:
            Could use higher level function "reduce" but might be slower

        IMPORTANT NOTES:
            ln(0) is Undefined so only use ln for non 0 probabilities

        :param list_word:
        :param text_classification_container:
        :param laplace_smoothing:
        :return:
        """
        result = text_classification_container.probability_prior

        if laplace_smoothing:
            result = math.log(result, math.e)

        # print(f"result {result}")

        for word in list_word:
            # print(word)
            if not laplace_smoothing:
                probability_word = text_classification_container.dict_k_word_v_frequency.get(word, 0)
            else:

                """
                This can only handle words in the both training sets, if a given word does not exist in either training
                set then this will return None because the natural log of 0 (0 probably of that word in the dataset)
                is Undefined.
                """
                probability_word = text_classification_container.dict_k_word_v_frequency_laplace_smoothing.get(word)

            # print(f"probability_word {probability_word}")

            if laplace_smoothing:

                # *** This try except will handle if probability_word == 0 as in the word does not exist in the dataset
                try:
                    result += math.log(probability_word, math.e)
                except Exception as e:
                    # traceback.print_exc()
                    # print(e)
                    pass
            else:
                result *= probability_word

            # print(f"result {result}")

        return result

    def classify_text(self, text, laplace_smoothing: bool = False):
        """
        Based on the classification dataset, calculate the probabilities for each classification and classify the text

        :param text:
        :param laplace_smoothing:
        :return:
        """
        classification_result = None

        list_tuple_v_classification_v_probability = []

        for classification, text_classification_container in self.dict_k_classification_v_text_classification_container.items():
            # print(classification)

            words = filter_text_to_words_v2(text)

            probability = self._classify_text_helper(words, text_classification_container, laplace_smoothing)

            list_tuple_v_classification_v_probability.append([classification, probability])

            # print()

        list_tuple_v_classification_v_probability.sort(key=lambda i: i[1], reverse=True)

        # print(list_tuple_v_classification_v_probability)

        # Determine the result of the classification of the given text
        if sum(i[1] for i in list_tuple_v_classification_v_probability) != 0:
            classification_result = list_tuple_v_classification_v_probability[0][0]

        # print(classification_result)
        return list_tuple_v_classification_v_probability, classification_result

    def print_all_classify_text(self, text):
        """
        Run the Naive Bayes algorithm with and without laplace smoothing

        :param text: text given by the user
        :return: list of classifications in the order of the algorithms ran
        """
        print("Text: \"{}\"\n".format(text))

        dict_k_classification_v_correct_count = {}

        result_standard = self.classify_text(text)
        self._print_all_classify_text_helper("Naive Bayes (Standard)",
                                             result_standard)

        result_laplace_smoothing = self.classify_text(text, laplace_smoothing=True)
        self._print_all_classify_text_helper("Naive Bayes (Laplace smoothing, alpha == {}, with ln)".format(self.alpha),
                                             result_laplace_smoothing)

        return result_standard[1], result_laplace_smoothing[1]

    @staticmethod
    def _print_all_classify_text_helper(classify_algorithm_name: str, result: Tuple[list, str]):
        """
        Helper printer for print_all_classify_text

        :param classify_algorithm_name:
        :param result:
        :return:
        """

        list_given = result[0]
        classification = result[1]

        print("\tClassification algorithm name: {}".format(classify_algorithm_name))

        if classification is None:
            print("\t\tCannot classify text, is the dataset too small?")
        else:
            print("\t\tThe text is probably: \"{}\"".format(list_given[0][0]))

        for classification, probability in list_given:
            print("\t\t\t\"{}\" probability: {}".format(classification, probability))
        print()


def run_example():
    """
    Run example of Covid News vs Non Covid News

    Notes:
        The first 4 examples in list_tuple are what I thought of
        The remaining are headers from news sites

    :return: None
    """
    file_text_news_covid = "text_news_covid"
    file_text_news_covid_not = "text_news_covid_not"

    dict_k_class_v_file_abs_path = {"Covid News": file_text_news_covid,
                                    "Non Covid News": file_text_news_covid_not}

    naive_bayes_object = NaiveBayes(dict_k_class_v_file_abs_path)

    naive_bayes_object.print_text_classification_container()
    print("#" * 100)
    print()

    list_tuple = [
        ("Covid", "Covid News"),
        ("The Royal Family", "Non Covid News"),
        ("You should get a test", "Covid News"),
        ("How is your marriage", "Non Covid News"),
        ("He was the murderer", "Non Covid News"),
        ("You have covid", "Covid News"),
        ("Buckingham Palace says royal family is 'saddened' in first statement since Harry and Meghan interview",
         "Non Covid News"),
        ("Covid relief package marks a shift in US battle against poverty",
         "Covid News"),
        ("The pandemic forced a massive remote-work experiment. Now comes the hard part",
         "Covid News"),
        ("End-of-life doulas help people die comfortably. In a pandemic, they're more important than ever",
         "Covid News"),
        ("Heavy rainfall in Hawaii's Maui damaged homes, sparked evacuations and worries over possible dam failure",
         "Non Covid News"),
        ("Jury selection begins in Derek Chauvin's trial in the death of George Floyd. Here's what to expect",
         "Non Covid News"),
        ("Rare meteorite that fell on UK driveway may contain 'ingredients for life'",
         "Non Covid News")
    ]

    count_classification_correct_standard = 0
    count_classification_correct_laplace = 0

    for text, classification in list_tuple:
        list_classification = naive_bayes_object.print_all_classify_text(text)
        if list_classification[0] == classification:
            count_classification_correct_standard += 1
            print("The Naive Bayes (Standard) algorithm is Correct in classifying the text")
        else:
            print("The Naive Bayes (Standard) algorithm is Incorrect in classifying the text")

        if list_classification[1] == classification:
            count_classification_correct_laplace += 1
            print("The Naive Bayes (Laplace smoothing) algorithm is Correct in classifying the text")
        else:
            print("The Naive Bayes (Laplace smoothing) algorithm is Incorrect in classifying the text")

        print()
        print("#" * 100)
        print()

    print("The Naive Bayes (Standard) algorithm has classified {} out of {} correct!".format(
        count_classification_correct_standard,
        len(list_tuple)))
    print("The Naive Bayes (Laplace smoothing) algorithm has classified {} out of {} correct!".format(
        count_classification_correct_laplace, len(list_tuple)))
    print()


def ask_user_loop():
    """
    Apparently, you need to ask the user so here is a loop for that...

    :return:
    """
    file_text_news_covid = "text_news_covid"
    file_text_news_covid_not = "text_news_covid_not"

    dict_k_class_v_file_abs_path = {"Covid News": file_text_news_covid,
                                    "Non Covid News": file_text_news_covid_not}

    naive_bayes_object = NaiveBayes(dict_k_class_v_file_abs_path)

    naive_bayes_object.print_text_classification_container()

    while True:
        text = input("Enter text: ")

        if text == "exit":
            return

        naive_bayes_object.print_all_classify_text(text)
        print()


if __name__ == '__main__':
    run_example()

    # print("#"*200)
    # ask_user_loop()
