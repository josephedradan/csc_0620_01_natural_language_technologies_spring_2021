"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 5/4/2021

Purpose:
    Do lessons 1 to 5 for Submission for hands-on workshop on Apr 29th Assignment

Details:

Description:

Notes:

    10 impressive applications of deep learning methods in the field of natural language processing
        Learning to Summarize with Human Feedback
            https://openai.com/blog/learning-to-summarize-with-human-feedback/
                Article summarization

        Language Models are Few-Shot Learners
            https://arxiv.org/abs/2005.14165
                GPT 3

        ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS
            https://openreview.net/attachment?id=H1eA7AEtvS&name=original_pdf
                Albert

        Reformer: The Efficient Transformer
            https://arxiv.org/abs/2001.04451
                Lite transformer

        XLNet: Generalized Autoregressive Pretraining for Language Understanding
            https://arxiv.org/abs/1906.08237

        Megatron-LM
            https://github.com/NVIDIA/Megatron-LM
                Big transformer

        Conversational AI
            https://rasa.com/open-source/
                Make a chat bot

        ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators
            https://arxiv.org/abs/2003.10555
                Alternative training solution to Masked language modeling

        Longformer: The Long-Document Transformer
            https://arxiv.org/abs/2004.05150
                Process large corpora

        BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
            https://arxiv.org/abs/1910.13461
                A Transformer

IMPORTANT NOTES:

Explanation:

Reference:
    How to Get Started with Deep Learning for Natural Language Processing
        Notes:
            The assigment
        Reference:
                https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/

TODO: REDO ENTIRE THING IN JOSEPH TESTS

"""

import nltk
import numpy as np
import os
import pandas
from gensim.models import Word2Vec
from itertools import chain
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def lesson_2():
    """
    Your task is to locate a free classical book on the Project Gutenberg website, download the ASCII version of the
    book and tokenize the text and save the result to a new file. Bonus points for exploring both manual and NLTK
    approaches.

    Post your code in the comments below. I would love to see what book you choose and how you chose to tokenize it.

    Reference:
        2. Accessing Text Corpora and Lexical Resources
            http://www.nltk.org/book/ch02.html

    :return:
    """
    files = nltk.corpus.gutenberg.fileids()

    list_word_objects = [nltk.corpus.gutenberg.words(i) for i in files]

    words = chain(*list_word_objects)

    list_str_files = [nltk.corpus.gutenberg.raw(i) for i in files]

    string = " ".join(list_str_files)

    if not os.path.exists("words.txt"):
        print("Writing to words.txt")
        with open("words.txt", "w", encoding="UTF-8") as f:
            iterable_token = nltk.tokenize.word_tokenize(string)

            f.write(" ".join(iterable_token))


def lesson_3():
    """
    Your task in this lesson is to experiment with the scikit-learn and Keras methods for encoding small contrived
    text documents for the bag-of-words model. Bonus points if you use a small standard text dataset of documents
    to practice on and perform data cleaning as part of the preparation.

    Post your code in the comments below. I would love to see what APIs you explore and demonstrate.

    Reference:
        Code
            https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/

    :return:
    """

    files = nltk.corpus.gutenberg.fileids()

    list_list_word = [nltk.corpus.gutenberg.sents(i) for i in files]

    list_list_sentence = [z for i in list_list_word for z in i]

    sentences = [" ".join(i) for i in list_list_sentence]

    vectorizer_tfid = TfidfVectorizer()

    vectorizer_tfid.fit(sentences)

    vector_sentence = vectorizer_tfid.transform([sentences[42]])

    print("Bag-of-Words with scikit-learn")
    print()
    # print(vectorizer_tfid.vocabulary_)
    print("vectorizer_tfid.idf_")
    print(vectorizer_tfid.idf_)
    print()
    print("sentences[42]")
    print(sentences[42])
    print()
    print("vector_sentence.shape")
    print(vector_sentence.shape)
    print()
    print("vector_sentence.toarray()")
    print(vector_sentence.toarray())
    print()
    print("=" * 50)
    print()
    print("Bag-of-Words with Keras")
    print()

    keras_tokenizer = Tokenizer()
    keras_tokenizer.fit_on_texts(sentences)

    print("keras_tokenizer.word_counts")
    print(pandas.DataFrame(list(keras_tokenizer.word_counts)))
    print()
    print("keras_tokenizer.document_count")
    print(keras_tokenizer.document_count)
    print()
    print("keras_tokenizer.word_index")
    print(pandas.DataFrame(list(keras_tokenizer.word_index)))
    print()
    print("keras_tokenizer.word_docs")
    print(pandas.DataFrame(list(keras_tokenizer.word_docs)))


def display_pca_scatterplot(model, words=None, sample=0):
    """

    Reference:
        Gensim word vector visualization of various word vectors
            Notes:
                I modified their code to be modern
            Reference:
                https://web.stanford.edu/class/cs224n/materials/Gensim%20word%20vector%20visualization.html

    :param model:
    :param words:
    :param sample:
    :return:
    """
    if words == None:
        if sample > 0:
            # words is a list of random words
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            # words is all the words (DANGEROUS)
            words = [word for word in model.vocab]

    # Get the vectors of the words in an np array
    word_vectors = np.array([model[w] for w in words])

    # Get the Principal Component Analaysis data (Select 2 PCAs that represent the data in the best way)
    twodim = PCA().fit_transform(word_vectors)[:, :2]

    # Plot teh PCA
    plt.figure(figsize=(16, 16))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors='k', c='r')

    for word, (x, y) in zip(words, twodim):
        plt.text(x + 0.05, y + 0.05, word)

    plt.savefig('lesson_3.png')

    plt.show()


def lesson_4():
    """
    Your task in this lesson is to train a word embedding using Gensim on a text document, such as a book from Project
    Gutenberg. Bonus points if you can generate a plot of common dict_words.

    Post your code in the comments below. I would love to see what book you choose and any details of the embedding that
    you learn.

    Reference:
        Gensim word vector visualization of various word vectors
            https://web.stanford.edu/class/cs224n/materials/Gensim%20word%20vector%20visualization.html

        models.word2vec – Word2vec embeddings
            https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.wv
    :return:
    """

    files = nltk.corpus.gutenberg.fileids()

    list_list_word = [nltk.corpus.gutenberg.sents(i) for i in files]

    list_list_sentence = [z for i in list_list_word for z in i]

    sentences = [" ".join(i) for i in list_list_sentence]

    """
    min_count:
        Ignores all words with total frequency lower than this.
    """
    gensim_word2vec_model = Word2Vec(list_list_sentence, min_count=1)
    print("gensim_word2vec_model")
    print(gensim_word2vec_model)
    print()
    dict_words = gensim_word2vec_model.wv.key_to_index
    print("list(gensim_word2vec_model.wv.key_to_index) (formerly gensim_word2vec_model.wv.vocab)")
    print(pandas.DataFrame(list(dict_words)))
    print()
    print("dict_words['sentence']")
    print(dict_words['sentence'])
    print()

    gensim_model_wv = gensim_word2vec_model.wv
    # print(model_wv["sentence"])  # Vector

    print("gensim_word2vec_model.wv.index_to_key")
    print(pandas.DataFrame(gensim_word2vec_model.wv.index_to_key))  # List word
    print()
    display_pca_scatterplot(gensim_model_wv, gensim_word2vec_model.wv.index_to_key[0:100])


def lesson_5_easy():
    """
    Your task in this lesson is to design a small document classification problem with 10 documents of one sentence
    each and associated labels of positive and negative outcomes and to train a network with word embedding on these
    data. Note that each sentence will need to be padded to the same maximum length prior to training the model using
    the Keras pad_sequences() function. Bonus points if you load a pre-trained word embedding prepared using Gensim.

    Post your code in the comments below. I would love to see what sentences you contrive and the skill of your model.

    IMPORTANT NOTES:
        REQUIRES THE BELOW IN THE SAME DIR
            https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

    :return:
    """

    # # define problem
    # vocab_size = 100
    # max_length = 32
    #
    # # define the model (1 hidden layer model)
    # keras_model = Sequential()
    # keras_model.add(Embedding(vocab_size, 8, input_length=max_length))
    # keras_model.add(Flatten())
    # keras_model.add(Dense(1, activation='relu'))  # Activation function
    #
    # # compile the model
    # keras_model.compile(optimizer='adam',  # alternative to Gradient Decent
    #                     loss='binary_crossentropy',  # Loss function to use at the end
    #                     metrics=['accuracy'])  #
    #
    # # summarize the model
    # print(keras_model.summary())

    filepath_dict = {'yelp': 'sentiment labelled sentences/yelp_labelled.txt',
                     'amazon': 'sentiment labelled sentences/amazon_cells_labelled.txt',
                     'imdb': 'sentiment labelled sentences/imdb_labelled.txt'}

    df_list = []
    for source, filepath in filepath_dict.items():
        df = pandas.read_csv(filepath, names=['sentence', 'label'], sep='\t')
        df['source'] = source  # Add another column filled with the source name
        df_list.append(df)

    df = pandas.concat(df_list)

    for source in df['source'].unique():
        df_source = df[df['source'] == source]
        sentences = df_source['sentence'].values
        y = df_source['label'].values

        sentences_train, sentences_test, y_train, y_test = train_test_split(
            sentences, y, test_size=0.25, random_state=1000)

        vectorizer = CountVectorizer()
        vectorizer.fit(sentences_train)

        X_train = vectorizer.transform(sentences_train)
        X_test = vectorizer.transform(sentences_test)

        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        print('Accuracy for {} data: {:.4f}'.format(source, score))

        print(sentences_test[0])
        print(classifier.predict(X_test[0]))
        print()


def lesson_5():
    """
    Your task in this lesson is to design a small document classification problem with 10 documents of one sentence
    each and associated labels of positive and negative outcomes and to train a network with word embedding on these
    data. Note that each sentence will need to be padded to the same maximum length prior to training the model using
    the Keras pad_sequences() function. Bonus points if you load a pre-trained word embedding prepared using Gensim.

    Post your code in the comments below. I would love to see what sentences you contrive and the skill of your model.

    IMPORTANT NOTES:
        REQUIRES THE BELOW IN THE SAME DIR
            https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

    Reference:
        Practical Text Classification With Python and Keras
            https://realpython.com/python-keras-text-classification/
    :return:
    """


if __name__ == '__main__':
    lesson_2()
    print("\n" + "#" * 100 + "\n")
    lesson_3()
    print("\n" + "#" * 100 + "\n")
    lesson_4()
    print("\n" + "#" * 100 + "\n")
    lesson_5_easy()
    print("\n" + "#" * 100 + "\n")
    lesson_5()
