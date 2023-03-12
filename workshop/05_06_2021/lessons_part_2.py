"""
Created by Joseph Edradan
Github: https://github.com/josephedradan

Date created: 5/11/2021

Purpose:
    Do lessons 6 to 7 for Submission for hands-on workshop on May 6th Assignment

Details:

Description:

Notes:

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
from collections import Counter
from os import listdir
from string import punctuation

import keras
import numpy as np
import os
from nltk.corpus import stopwords


def lesson_5_redone():
    """
    Your task in this lesson is to design a small document classification problem with 10 documents of one sentence
    each and associated labels of positive and negative outcomes and to train a network with word embedding on these
    data. Note that each sentence will need to be padded to the same maximum length prior to training the model using
    the Keras pad_sequences() function. Bonus points if you load a pre-trained word embedding prepared using Gensim.

    Reference:
        How to One Hot Encode Sequence Data in Python
            https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

        tf.keras.preprocessing.text.one_hot
            https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/one_hot
    :return:
    """

    # define documents
    docs = ['Well done!',
            'Good work',
            'Great effort',
            'nice work',
            'Excellent!',
            'Weak',
            'Poor effort!',
            'not good',
            'poor work',
            'Could have done better.']

    # define class labels
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    # integer encode the documents

    vocab_size = 50
    encoded_docs = [keras.preprocessing.text.one_hot(d, vocab_size) for d in docs]
    print("+" * 50)
    print("encoded_docs")
    print(encoded_docs)  # Encoding is random
    print()

    # pad documents to a max length of 4 words (Makes encoding of words a size of 4)
    max_length = 4
    padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print("+" * 50)
    print("padded_docs")
    print(padded_docs)
    print()

    # define the model
    model = keras.Sequential()  # Sequence model
    model.add(keras.layers.embeddings.Embedding(vocab_size, 8, input_length=max_length))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # summarize the model
    print("+" * 50)
    print("model.summary()")
    print(model.summary())
    print()

    # fit the model
    model.fit(padded_docs, labels, epochs=50, verbose=0)

    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
    print("+" * 50)
    print('Accuracy: %f' % (accuracy * 100))
    print()


def lesson_6():
    """
    Your task in this lesson is to research the use of the Embeddings + CNN combination of deep learning methods for
    text classification and report on examples or best practices for configuring this model, such as the number of
    layers, kernel size, vocabulary size and so on.

    Bonus points if you can find and describe the variation that supports n-gram or multiple groups of words as input
    by varying the kernel size.

    Post your findings in the comments below. I would love to see what you discover.

    Notes CNN:
        Kernel is the filter
        Max pooling, select the largest number from the filter
        Epoch amount is the amount of times a forward and backward pass has happened on the entire dataset
        Batch size is the sample size of the entire dataset. Training dataset/batch_size == iterations

    Notes:
        1.
            neural networks in general offer better performance than classical linear classifiers, especially
            when used with pre-trained word embeddings.

            convolutional neural networks are effective at document classification, namely because they are able to
            pick out salient (important) features (e.g. tokens or sequences of tokens) in a way that is invariant to
            their position within the input sequences (find specific important words in sentences).

            "Convolutional and pooling layers allow the model to learn to find such local indicators,
            regardless of their position."

            Word Embedding: A distributed representation of words where different words that have a similar meaning
            (based on their usage) also have a similar representation.
            Convolutional Model: A feature extraction model that learns to extract salient features
            from documents represented using a word embedding.
            Fully Connected Model: The interpretation of extracted features in terms of a predictive output.

            "… the CNN is in essence a feature-extracting architecture. It does not constitute a standalone,
            useful network on its own, but rather is meant to be integrated into a larger network, and to be
            trained to work in tandem with it in order to produce an end result. The CNNs layer’s responsibility
             is to extract meaningful sub-structures that are useful for the overall prediction task at hand."

            Key:
                Use CNN on Word embedding
                CNN extracts important words

        2.
            Transfer function: rectified linear.
            Kernel sizes: 3, 4, 5.
            Number of filters: 100
            Dropout rate: 0.5
            Weight regularization (L2): 3
            Batch Size: 50
            Update Rule: Adadelta

            tuning of the word vectors offer a small additional improvement in performance.

            Key:
                Small batch size,
                Kernel size: 3 to 5
                Train with 50 samples at a time


        3.
            Some hyperparameters matter more than others when tuning a convolutional neural network on your document
            classification problem.

            The choice of pre-trained word2vec and GloVe embeddings differ from problem to problem,
            and both performed better than using one-hot encoded word vectors.
            The size of the kernel is important and should be tuned for each problem.
            The number of feature maps is also important and should be tuned.
            The 1-max pooling generally outperformed other types of pooling.
            Dropout has little effect on the model performance.

            Use word2vec or GloVe word embeddings as a starting point and tune them while fitting the model.
            Grid search across different kernel sizes to find the optimal configuration for your problem, in the range
            1-10.
            Search the number of filters from 100-600 and explore a dropout of 0.0-0.5 as part of the same search.
            Explore using tanh, relu, and linear activation functions.

            Data is based on binary text classification problems using single sentences as input.

            Key:
                Kernel size needs to be turned for the problem
                Don't care about drop out

        4.
            "ConvNets do not require the knowledge about the syntactic or semantic structure of a language."
            "Working on only characters also has the advantage that abnormal character combinations such as
            misspellings and emoticons may be naturally learnt."

            The model achieves some success, performing better on problems that offer a larger corpus of text.

            Key:
                Use letters than words

        5.
            Use letters than words

            5 to 6 layers of CNN is small
            "we propose to use deep architectures of many convolutional layers to approach this goal,
            using up to 29 layers."

            "We present a new architecture (VDCNN) for text processing which operates directly at the character level
            and uses only small convolutions and pooling operations."

            The very deep architecture worked well on small and large datasets.
            Deeper networks decrease classification error.
            Max-pooling achieves better results than other, more sophisticated types of pooling.
            Generally going deeper degrades accuracy; the shortcut connections used in the architecture are important.

            Key:
                Use letters and more layers, but not too much layers

        Over all key:
            1. Convolutional neural networks are effective at document classification, namely because they are able to
            pick out salient (important) features.
            2. Deeper models that operate directly on text may be the future of natural language processing.
            3. Use letters than words OR use Word embeddings
            4. Kernel filter size for CNN should be testing from 1 to 10


    Reference:
        Best Practices for Text Classification with Deep Learning
            Reference:
                https://machinelearningmastery.com/best-practices-document-classification-deep-learning/

        8. Text Classification Using Convolutional Neural Networks
            Reference:
                https://www.youtube.com/watch?v=8YsZXTpFRO0

            Notes:
                CNN are fixed size
                Amount of channels are based on the amount of vector components a word embedding has
                Kernel size 3
                Embedding layer's dimensions == channels



    :return:
    """
    # define problem
    vocab_size = 100
    max_length = 200
    # define model
    model = keras.Sequential()
    model.add(keras.layers.embeddings.Embedding(vocab_size, 100, input_length=max_length))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    print(model.summary())

    model.compile(loss='Binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# turn a doc into clean tokens
def clean_doc_train(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens


# save list to file
def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # print(tokens)

    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


# load all docs in a directory
def process_docs(directory, vocab):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # print(filename)
        # skip files that do not have the right extension

        if not filename.endswith(".txt"):
            continue

        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines


# load all docs in a directory
def process_docs_train(directory, vocab, is_trian):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc_train(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents


# load all docs in a directory
def process_docs_save(directory, vocab):
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path, vocab)


# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)


def lesson_7():
    """
    Your task in this lesson is to develop and evaluate a deep learning model on the movie review dataset:

    1. Download and inspect the dataset.
    2. Clean and tokenize the text and save the results to a new file.
    3. Split the clean data into train and test datasets.
    4. Develop an Embedding + CNN model on the training dataset.
    5. Evaluate the model on the test dataset.
    Bonus points if you can demonstrate your model by making a prediction on a new movie review, contrived or real.
    Extra bonus points if you can compare your model to a neural bag-of-words model.

    Post your code and model skill in the comments below. I would love to see what you can come up with.
    Simpler models are preferred, but also try going really deep and see what happens.


    Reference:
        How to Prepare Movie Review Data for Sentiment Analysis (Text Classification)
            https://machinelearningmastery.com/prepare-movie-review-data-sentiment-analysis/

        Deep Convolutional Neural Network for Sentiment Analysis (Text Classification)
            https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/

        Model saving & serialization APIs
            https://keras.io/api/models/model_saving_apis/

    :return:
    """

    if not os.path.exists('vocab.txt'):
        #####

        # define vocab
        vocab = Counter()

        # add all docs to vocab
        process_docs_save('txt_sentoken/neg', vocab)
        process_docs_save('txt_sentoken/pos', vocab)

        # print the size of the vocab
        print(len(vocab))

        # print the top words in the vocab
        print(vocab.most_common(50))

        # keep tokens with > 5 occurrence
        min_occurane = 5
        tokens = [k for k, c in vocab.items() if c >= min_occurane]
        print(len(tokens))

        # save tokens to a vocabulary file
        save_list(tokens, 'vocab.txt')

        #####

    # load vocabulary
    vocab_filename = 'vocab.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()  # Split by space
    vocab = set(vocab)

    # MEANT FOR THE ORIGINAL PROCESSING, DON'T USE FOR KERAS
    # prepare negative reviews

    if not os.path.exists('negative.txt'):
        negative_docs = process_docs('txt_sentoken/neg', vocab)
        save_list(negative_docs, 'negative.txt')

    # # prepare positive reviews

    if not os.path.exists('positive.txt'):
        positive_docs = process_docs('txt_sentoken/pos', vocab)
        save_list(positive_docs, 'positive.txt')

    #######################################################

    # load all training reviews
    positive_docs = process_docs_train('txt_sentoken/pos', vocab, True)
    negative_docs = process_docs_train('txt_sentoken/neg', vocab, True)
    train_docs = negative_docs + positive_docs

    # create the tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)

    # sequence encode
    encoded_docs = tokenizer.texts_to_sequences(train_docs)

    # pad sequences
    max_length = max([len(s.split()) for s in train_docs])
    Xtrain = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    # define training labels
    ytrain = np.array([0 for _ in range(900)] + [1 for _ in range(900)])

    # load all test reviews
    positive_docs = process_docs_train('txt_sentoken/pos', vocab, False)
    negative_docs = process_docs_train('txt_sentoken/neg', vocab, False)
    test_docs = negative_docs + positive_docs

    # sequence encode
    encoded_docs = tokenizer.texts_to_sequences(test_docs)

    # pad sequences
    Xtest = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    # define test labels
    ytest = np.array([0 for _ in range(100)] + [1 for _ in range(100)])

    # define vocabulary size (largest integer value)
    vocab_size = len(tokenizer.word_index) + 1

    # If the model does not exist
    if not os.path.exists("model.h5"):
        # define model
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(vocab_size, 100, input_length=max_length))
        model.add(keras.layers.convolutional.Conv1D(filters=32, kernel_size=8, activation='relu'))
        model.add(keras.layers.convolutional.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(10, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        print(model.summary())

        # compile network
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit network
        model.fit(Xtrain, ytrain, epochs=10, verbose=2)

        # Save model
        model.save("model.h5")
        print("Model saved")
        print()

    # Load model
    model = keras.models.load_model("model.h5")

    # evaluate
    loss, acc = model.evaluate(Xtest, ytest, verbose=0)
    print('Test Accuracy: %f' % (acc * 100))


if __name__ == '__main__':
    print("\n" + "#" * 150 + "\n")
    print("Lesson 5 redone")
    print("\n" + "#" * 150 + "\n")
    lesson_5_redone()

    print("\n" + "#" * 150 + "\n")
    print("Lesson 6")
    print("\n" + "#" * 150 + "\n")
    lesson_6()

    print("\n" + "#" * 150 + "\n")
    print("Lesson 7")
    print("\n" + "#" * 150 + "\n")
    lesson_7()
