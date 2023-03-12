#!/usr/bin/env python
# coding: utf-8

# ## CSC 0620-01 Natural Language Technologies Spring 2021
# 
# Joseph Edradan <br>
# 03/23/2021 <br>
# 
# #### Submission for hands-on workshop on Mar 18th
# 
# <ol>
#   <li>Understand the Logistic Regression based sentiment analysis implemented here: <a href="https://www.kaggle.com/davidoluwatobipeter/sentiment-analysis-logistic-regression-and-lstm">Sentiment Analysis: Logistic Regression and LSTM
# </a></li>
#   <li>Create a copy of the above Jupyter notebook (or export it as python program).  In this new notebook (or program) add a detailed description (and in your own words) of what is happening in each code block. You need to implement only till the 62nd step as the rest involves LSTM model and fine-tuning LR model.
# </li>
# </ol>

# Classification is the process of predicting the class of given data points. Classes are sometimes called targets/ labels or categories. Classification predictive modeling is the task of approximating a mapping function (f) from input variables (X) to discrete output variables (y). Classification belongs to the category of supervised learning where the targets also provided with the input data.
# 
# A classifier utilizes some training data to understand how given input variables relate to the class.

# Classification is the process of predicting the class of given data points. Classes are sometimes called targets/ labels or categories. Classification predictive modeling is the task of approximating a mapping function (f) from input variables (X) to discrete output variables (y). Classification belongs to the category of supervised learning where the targets also provided with the input data.
# 
# A classifier utilizes some training data to understand how given input variables relate to the class.

# This end goal of this project is to build models - Logistic Regression, a classical traditional machine learning algorithm and Long Short Term Memory, a deep learning algorithm that can predict the sentiment category or class of texts based on covid19 related tweets
# 
# This problem is an instance of multiclass classification; and because each data point should be classified into only one category, the problem is more specifically an instance of single-label, multiclass classification.
# 
# The dataset used for this project can be found at https://www.kaggle.com/datatattle/covid-19-nlp-text-classification?rvi=1

# Python Libraries Needed for File Opening, Data Analysis, Data Visualization, Data Exploration and Data Cleaning

# In[1]:


"""
Import libraries and use a matplotlib magic function to put the plots inline

"""
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


"""
Names for the dataset
"""
train = 'Corona_NLP_train.csv'
test = 'Corona_NLP_test.csv'


# In[3]:


# A copy of the files is preserved inorder to retain an original structure which won't be affected by
# the data processings.


# In[4]:


"""
Mand pandas DataFrame with latin-1 beecause of special characters
"""
trainOriginal = pd.read_csv(train, encoding='latin-1')
testOriginal = pd.read_csv(test, encoding='latin-1')


# In[5]:


"""
Make copies of the DataFrames
"""
train = trainOriginal.copy()
test = testOriginal.copy()


# In[6]:


# Earlier attempts to open the files with a utf-8 encoding lead to a unicode error as it couldn't
# parse certain parts of the file, hence, utf-8 was introduced as a solution.


# In[7]:


"""
Show first 5 of the training dataset
"""
train.head()


# In[8]:


"""
Show first 5 of the training dataset
"""
test.head()


# In[9]:


# The datasets contain 7 columns housing the data info. The UserName and ScreenName has being
# encrypted due to privacy concerns. The tweets contains mentions and hashtags which must be cleaned
# in order to help the models better understand the statistical relationship between the relevant
# details. The sentiment column contains 5 different classes which can be remapped into 3 for better
# statistical understanding. The other columns are the timeframe of the tweets and the location from
# where the tweets where twitted.


# In[10]:


"""
Show training dataset info
"""
train.info()


# In[11]:


# The datas that will have the major effects on how the models determine the classes are non integers
# which means data preprocessing steps needs to be don before feeding them into the models.


# In[12]:


"""
Show training dataset empty cell
"""
train.isnull().sum()


# In[13]:


# The location column contains a whooping 8590 missing rows. Filling the blanks with the most common
# location won't really make sense as the missing details are too much.


# In[14]:


"""
Show training dataset's Location column from row 0 to 60
"""
train['Location'].value_counts()[:60]


# In[15]:


# The location info was splitted and merged because lots of the locations are in the same geolocation,
# The datasets also shows a great reprentative bias. For a global distortion, the bulk of the data
# collected are within few geopolitical zones with Africa having a little representation. Different
# ideals, govermental polices, religious beliefs etc are factors that likely influenced the sentiment
# class of individual tweets.


# In[16]:


"""
In both the training and testing dataset split cell's string by a delimiter and use only the first string split by the delimiter
"""
# splitting location into word pairs
train['Location'] = train['Location'].str.split(",").str[0]
test['Location'] = test['Location'].str.split(",").str[0]


# In[17]:


"""
Show training dataset's Location column from row 0 to 60 (should be clean now)
"""
train['Location'].value_counts()[:60]


# In[18]:


"""
Show training dataset's TweetAt count of the same cell data (Count the amount of times a cell has the same value)
"""
train['TweetAt'].value_counts()


# In[19]:


# The data collected was tweeted between 16th March, 2020 to 14th April, 2020. Any model built and
# deployed at this time may likely not be relevant for present use due to new findings, researches,
# tresnd that have emerged which will influence every recent covid19 related tweets. Any model built
# using this data will be a decayed model and further decay will happen at a rapid pace.


# In[20]:


"""
Show training dataset's Sentiment count of the same cell data (Count the amount of times a cell has the same value)
"""
train['Sentiment'].value_counts()


# In[21]:


# Positive > Negative > Neutral and the categories will be remapped to fully represent this position.


# Visual Representation of the Training Set

# In[22]:


"""
Use matplotlib to plot a seaborn (a library for making colorful representations of data) count plot on the location count 
"""
plt.figure(figsize=(10, 10))
sns.countplot(y='Location', data=train, order=train.Location.value_counts().iloc[
    0:19].index).set_title("Twitted locations")


# In[23]:


"""
Use matplotlib to plot a seaborn (a library for making colorful representations of data) count plot on the Sentiment count 
"""
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize': (11, 8)})
sns.countplot(train['Sentiment'])


# In[24]:


"""
Use matplotlib to plot pie plot on the Sentiment count 
"""
labels = ['Positve', 'Negative', 'Neutral',
          'Extremely Positive', 'Extremely Negative']
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff5645']
explode = (0.05, 0.05, 0.05, 0.05, 0.05)
plt.pie(train.Sentiment.value_counts(), colors=colors, labels=labels,
        autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode=explode)
centreCircle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centreCircle)
plt.tight_layout()
plt.show()


# In[25]:


"""
Make a DataFrame of the training dataset using all rows and columns Location and Sentiment
"""
# [:,[2,5]] is the location and sentiment columns
plotDf = train.iloc[:, [2, 5]]
plotDf


# In[26]:


"""
Use matplotlib to plot a seaborn (a library for making colorful representations of data) count plot on the Sentiment count based on Location
"""
sns.set(rc={'figure.figsize': (15, 9)})
gg = train.Location.value_counts()[:5].index
plt.title('Sentiment Categories of the First 5 Top Locations',
          fontsize=16, fontweight='bold')
sns.countplot(x='Location', hue='Sentiment', data=plotDf, order=gg)


# In[27]:


# Reflecting the insight from train['Sentiment'].value_counts(), positive sentiment dominates the
# kind of tweets across the locations.


# Data Processing for Machine Learning Algorithms
# 
# Data processing deals with preparing the input data and targets before feeding them into a machine learning model. Many data-preprocessing and feature-engineering techniques are domain specific (for example, specific to text data).
# 
# Data preprocessing aims at making the raw data at hand more amenable to machine learning algorithms. This includes vectorization, normalization, handling missing values, and feature extraction.
# 
# Particular to this project is the need to convert the tweets into vector arrays and padded sequences before feeding it into the logistic regression and LSTM models respectively.

# Both the test and train set are concatenated together to easily preprocess both together.
# 
# Training set will have an identity of 0 while the test set will have 1

# In[28]:


"""
Set a value for the Identity column for the traning and testing dataset 
"""
train['Identity'] = 0
test['Identity'] = 1

"""
Make a new pd DataFrame with both the training and testing dataset and reset the indices for concating two DataFrames
"""
covid = pd.concat([train, test])
covid.reset_index(drop=True, inplace=True)


# In[29]:


covid.head()


# The 5 sentiment categories are regrouped into 3 for easy data analysis

# In[30]:


"""
In the covid DF replace 'Extremely Positive' with 'Positive' and 'Extremely Negative' with 'Negative'
"""
covid['Sentiment'] = covid['Sentiment'].str.replace(
    'Extremely Positive', 'Positive')
covid['Sentiment'] = covid['Sentiment'].str.replace(
    'Extremely Negative', 'Negative')


# The screen and username columns are dropped since they'll have no effect on the accuracy of the model.

# In[31]:


"""
In the covid DF replace Drop columns 'ScreenName' and 'Username'
"""
covid = covid.drop('ScreenName', axis=1)
covid = covid.drop('UserName', axis=1)
covid


# In[32]:


# The blank rows in the Location column would have being filled with Unknown if it would have had any
# significant impact on the objective of the project
# covid['Location'].fillna('Unknown', inplace=True)

# covid.isnull().sum() would have being used to check and confirm


# Visualizing the Concanated Data Set

# In[33]:


"""
Use matplotlib on seaborn countplot on the column 'Sentiment'
"""
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize': (11, 8)})
sns.countplot(covid['Sentiment'])


# In[34]:


"""
Use matplotlib pie plot on the column 'Sentiment'
"""
labels = ['Positve', 'Negative', 'Neutral']
colors = ['lightblue', 'lightsteelblue', 'silver']
explode = (0.1, 0.1, 0.1)
plt.pie(covid.Sentiment.value_counts(), colors=colors, labels=labels,
        shadow=300, autopct='%1.1f%%', startangle=90, explode=explode)
plt.show()


# In[35]:


"""
Use matplotlib on seaborn countplot on Location count for every location on 'Location'
"""
plt.figure(figsize=(10, 10))
sns.countplot(y='Location', data=train, order=train.Location.value_counts().iloc[
    0:19].index).set_title("Twitted locations")


# The sentiment categories are remapped into three so that the classifiers will be more accurate.
# 
# Neutral: 0, Positive: 1, Negative: 2

# In[36]:


"""
In pd DataFrame covid column 'Sentiment', replace words with a number with a mapping 
"""
covid['Sentiment'] = covid['Sentiment'].map(
    {'Neutral': 0, 'Positive': 1, 'Negative': 2})


# Further Data Processing and Analysis - top mentions and hashtags in the tweets are extracted and analyzed, after which they will be removed as well as the stop words just to make it easier for the models to discover the statistical relationship between the words.

# In[37]:


"""
In pd DataFrame covid column 'OriginalTweet', Use regex to get strings that start with '#' and don't have any space after it.
Basically get all hashtags

Then print df based on hashtag count
"""
hashTags = covid['OriginalTweet'].str.extractall(r"(#\S+)")
hashTags = hashTags[0].value_counts()
hashTags[:50]


# In[38]:


# As expected, the bulk of the tweets centres around covid19, it's other generic names, safety
# protocols as well as the different materials needed to weather through the tough times.


# In[39]:


"""
In pd DataFrame train column 'OriginalTweet', Use regex to get strings that start with '#' and don't have any space after it.
Basically get all hashtags.

Then print df based on hashtag count
"""
mentions = train['OriginalTweet'].str.extractall(r"(@\S+)")
mentions = mentions[0].value_counts()
mentions[:50]


# A python regex function to clean the tweets by removing hashtags, mentions, urls, digits and stop words.

# In[40]:


"""
Make a function that uses regex to remove links, @, hashtags, numbers, <STUFF> with a space
"""
import re


def clean(text):
    text = re.sub(r'http\S+', " ", text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = text.split()
    text = " ".join([word for word in text if not word in stopWord])

    return text


# In[41]:


"""
Import nltk and its stopwords

"""
import nltk
from nltk.corpus import stopwords


# In[42]:


# Stop words are high-frequency words like a, an, the, to and also that we sometimes want to filter
# out of a document before further processing. Stop words usually have little lexical content and
# do not hold much of a meaning.

# Below is a list of 25 example of semantically non-selective stop words: a, an, and, are, as, at,
# be, by, for, from, has, he, in, is, it, its, of, on, that, the, to, was, were, will, with.


# In[43]:


"""
Get a stopwords object

"""
stopWord = stopwords.words('english')


# In[44]:


"""
Clean the tweets in the df covid column 'OriginalTweet'

"""
covid['OriginalTweet'] = covid['OriginalTweet'].apply(lambda x: clean(x))


# In[45]:


"""
Show first 5 of the covid df

"""
covid.head()


# Features not needed for the predictions are dropped

# In[46]:


"""
Modify the DF to only show 3 specific columns

"""
covid = covid[['OriginalTweet', 'Sentiment', 'Identity']]
covid.head()


# Libraries and Frame Works Needed for Further Data Processing, Building a Logistic Regression Model and it's Evaluation

# In[47]:


"""
From nltk import a Steemer and a Lemmatizer aswell as word and sentence tokenizers

From sklearn import Logistic regression and and functions to train, score, report, and tune a model
"""
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


# Machine learning models finds statistical relations, therefore the data is tokenized and vectorized as part of the data preprocessing step.

# In[48]:


# Lemmatization is the algorithmic process of determining the lemma of a word based on its intended
# meaning. For example, in English, the verb “to walk” may appear as “walk,” “walked,” “walks,” or
# “walking.” The base form, “walk,” that one might look up in a dictionary, is called the lemma for
# the word.

# Tokenization is one of the simple yet basic concepts of natural language processing where texts are
# splitted into meaningful segments.

# Data vectorization deals with the turning of data into tensors. All inputs and targets in a machine
# learning model must be tensors of floating-point data (or, in specific cases, tensors of integers).


# In[49]:


"""
On the covid df, tokenize the text an put it into the 'Corpus' column
"""
covid['Corpus'] = [nltk.word_tokenize(text) for text in covid.OriginalTweet]

"""
Lemmatize the corpus column 
"""
lemma = nltk.WordNetLemmatizer()
covid.Corpus = covid.apply(
    lambda x: [lemma.lemmatize(word) for word in x.Corpus], axis=1)
covid.Corpus = covid.apply(lambda x: " ".join(x.Corpus), axis=1)


# In[50]:


"""
Show first 5 of the covid DF
"""
covid.head()


# The data sets are splitted back into training and test set

# In[51]:


"""
Remake the training and testing dataset where the 'Identity' column has 0 or 1 respectively
"""
train = covid[covid.Identity == 0]
test = covid[covid.Identity == 1]

"""
Drop the 'Identity' column in both datasets and reset the indices for the testing dataset
"""
train.drop('Identity', axis=1, inplace=True)
test.drop('Identity', axis=1, inplace=True)
test.reset_index(drop=True, inplace=True)


# In[52]:


"""
Show first 5 of the training DF
"""
train.head()


# In[53]:


"""
Show first 5 of the testing DF
"""
test.head()


# The train set is splitted to get a validation set.

# In[54]:


"""
For the training and testing DataFrames, make DataFrames of what is the input ('Corpus' column) and what is the ouput ('Sentiment' column) 
"""

XTrain = train.Corpus
yTrain = train.Sentiment

XTest = test.Corpus
yTest = test.Sentiment


"""
Split the training dataset (will overwrite old XTrain and yTrain) 
"""
XTrain, XVal, yTrain, yVal = train_test_split(
    XTrain, yTrain, test_size=0.2, random_state=42)


# In[55]:


"""
Show amount of data for each df
"""
XTrain.shape, XVal.shape, yTrain.shape, yVal.shape, XTest.shape, yTest.shape


# In[56]:


"""
Create a CountVectorizer object and vectorize XTrain, XVal, and XTest
"""
vectorizer = CountVectorizer(
    stop_words='english', ngram_range=(1, 2), min_df=5).fit(covid.Corpus)

XTrainVec = vectorizer.transform(XTrain)
XValVec = vectorizer.transform(XVal)
XTestVec = vectorizer.transform(XTest)


# Logistic Regression Performance

# In[57]:


# Logistic Regression (also called Logit Regression) is commonly used to estimate the probability
# that an instance belongs to a particular class (e.g., what is the probability that this email is
# spam?). If the estimated probability is greater than 50%, then the model predicts that the instance
# belongs to that class (called the positive class, labeled “1”), or else it predicts that it does
# not (i.e., it belongs to the negative class, labeled “0”).

# A Logistic Regression model computes a weighted sum of the input features (plus a bias term), but
# instead of outputting the result directly like the Linear Regression model does, it outputs the
# logistic of this result.
# p = hθ x = σ xTθ
# The logistic—noted σ(·)—is a sigmoid function (i.e., S-shaped) that outputs a number
# between 0 and 1.


# In[58]:


"""
Create a LogisticRegression object

"""
# solver='liblinear' fixes the decoding error
# logReg = LogisticRegression(random_state=42, solver='liblinear')
# logReg = LogisticRegression(random_state=42, max_iter=1000)
logReg = LogisticRegression(random_state=42)


# In[59]:


# Cross-validation makes it possible to get not only an estimate of the performance of models,
# but also a measure of how precise this estimate is (i.e., its standard deviation). But cross-
# validation comes at the cost of training models several times, so it is not always possible.


# In[60]:


# solver='liblinear' fixes the decoding error
# cross_val_score(LogisticRegression(random_state=42, solver='liblinear'),
#                 XTrainVec, yTrain, cv=10, verbose=1, n_jobs=-1).mean()

"""
Get the score of the LogisticRegression model to see how well it performs with a portion of XTrainVec to get a portion of yTrain

Notes:
    Use 10 fold cross validation (10 splits) to split the data into a training and testing dataset 
    then run the model to see how will it performed (10 times).
    Take the average of how well it performed at classifying the data.

"""

cross_val_score(LogisticRegression(random_state=42),
                XTrainVec, yTrain, cv=10, verbose=1, n_jobs=-1).mean()


# In[61]:


"""
Create a model using LogisticRegression Trained on XTrainVec and yTrain  
"""
model = logReg.fit(XTrainVec, yTrain)


# In[62]:


"""
Print how well the model did in predicting yVal given XValVec
"""
print(classification_report(yVal, model.predict(XValVec)))


# Fine Tuning the Logistic Regression Model
# 
# A great way to do this is by 'Grid Searching' which involves the fiddling with the hyperparameters until a great combination of hyperparameter values is discovered. It can be done simply by using Scikit-Learn’s GridSearchCV. All that's needed is tell it which hyperparameters you want it to experiment with, and what values to try out, and it will evaluate all the possible combinations of hyperparameter values, using cross-validation.

# In[63]:


"""
Setup the hyperparameters (paramters to 'guess' parameters for tuning the model)
"""
penalty = ['l2']
C = np.logspace(0, 4, 10)  # An arrry evenly spaced from 0 to 4 in log space with 10 samples
hyperparameters = dict(C=C, penalty=penalty)

"""
Use grid search cross validation to fine tune the given model logReg
"""
logRegGrid = GridSearchCV(logReg, hyperparameters, cv=5, verbose=0)


# In[64]:


"""
Train the "going to be fine tuned empty model of logReg" called logRegGrid with XTrainVec, yTrain
"""
bestModel = logRegGrid.fit(XTrainVec, yTrain)


# In[65]:


# Best hyperparameters combination

"""
Print the best combination of hyperparameters to tune the model

"""

print('Best Penalty:', bestModel.best_estimator_.get_params()['penalty'])
print('Best C:', bestModel.best_estimator_.get_params()['C'])


# In[66]:


# Final Logistic Regression model performance

"""
Run the fine tuned model on XTestVec to get yPred
"""
yPred = bestModel.predict(XTestVec)


# In[67]:


"""
Compare yPred to yTest
"""
print(classification_report(yTest, yPred))


# In[68]:


# Precision deals with the accuracy of the positive predictions.
# precision = TP / TP + FP
# TP is the number of true positives, and FP is the number of false positives.

# Recall, also called sensitivity or true positive rate (TPR) is the ratio of positive instances that
# are correctly detected by the classifier.
# recall = TP / TP + FN
# TP is the number of true positives FP is the number of false positives and FN is the number of
# false negatives.

# But the metric of choice to measure the performance of the logistic regression model in this
# project is the F1-score.The F1 score is the harmonic mean of precision and recall.
# Whereas the regular mean treats all values equally, the harmonic mean gives much more weight to low
# values. As a result, the classifier will only get a high F1 score if both recall and precision are
# high.


# In[69]:


# A less concise metric also available is the confusion matrix. The general idea involves counting
# the number of times instances of class A are classified as class B.

#  Implementation:

# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import cross_val_predict

# yPred = bestModel.predict(XTestVec)
# print(confusion_matrix(yTest, yPred))

# NB: it's possible that classification metrics wont't be able to handle a mix of multilabel-indicator
# and multiclass targets.


# ### LONG SHORT TERM MEMORY (LSTM) MODEL

# The underlying Long Short-Term Memory (LSTM) algorithm was developed by Hochreiter and Schmidhuber
# in 1997; it was the culmination of their research on the vanishing gradient problem. This layer is a variant of the SimpleRNN layer; it adds a way to carry information across many timesteps. Imagine a conveyor belt running parallel to the sequence you’re processing. Information from the sequence can jump onto the conveyor belt at any point, be transported to a later timestep, and jump off, intact,
# when you need it. This is essentially what LSTM does: it saves information for later, thus preventing older signals from gradually vanishing during processing (Francois Chollet, Deep Learning with Python).

# Libraries and Frame Works Needed for Further Data Processing, Building a LSTM Model and it's Evaluation

# In[70]:


from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dropout
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# In[71]:


lines = []
for line in train['Corpus']:
    lines.append(line)

len(lines)


# In[72]:


# Number of words to consider as features
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(train['Corpus'].values)
wordIndex = len(tokenizer.word_index) + 1
print('Found %s unique tokens.' % (wordIndex))


# In[73]:


# Turns the lists of integers into a 2D integer tensor of shape (numWords, maxlen)
XTrain = tokenizer.texts_to_sequences(train['Corpus'].values)
# Cuts off the texts after this number of words
XTrain = pad_sequences(XTrain, maxlen=30)

XTest = tokenizer.texts_to_sequences(test['Corpus'].values)
XTest = pad_sequences(XTest, maxlen=30)


# In[74]:


# The tokenizer selects the most common 5000 words. The sequences are padded so that they all have
# a uniform length of 30.


# In[75]:


XTrain.shape, XTest.shape


# In[76]:


yTrain = to_categorical(train['Sentiment'], 3)
yTest = to_categorical(test['Sentiment'], 3)


# NEURAL NETWORK

# The neural network consists of one embedding layer followed by one LSTM layer with 200 units. A Dropout layer is added for regularizatin to prevent overfitting of the model. The neural architecture ends with a Dense layer having three units to generate the output or prediction classes. The activation used is softmax since it is a single label, multi class problem.

# A prominent or distinguishing feature in this neural construct is the Embedding layer. The Embedding layer is best understood as a dictionary that maps integer indices (which stand for specific words) to dense vectors. It takes integers as input, it looks up these integers in an internal dictionary, and it returns the associated vectors. It’s effectively a dictionary lookup.

# In[77]:


model = models.Sequential()
model.add(layers.Embedding(wordIndex, 128, input_length=1000))
model.add(layers.LSTM(200))
model.add(Dropout(0.2))
model.add(layers.Dense(3, activation='softmax'))


# In[78]:


model.summary()


# In[79]:


model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=0.01),
              metrics=['accuracy'])


# In[80]:


# The callbacks parameter implemented monitors the validation loss and stops the training process
# once there is no apparent improvement for 10 epochs. It will also restore the best version of the
# model recorded during training.


# In[ ]:


history = model.fit(XTrain, yTrain, batch_size=250, epochs=100, validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])


# Visualizing the Loss/Accuracy of the Model In-Between Epochs

# In[ ]:


accuracy = history.history['accuracy']
valAccuracy = history.history['val_accuracy']
loss = history.history['loss']
valLoss = history.history['val_loss']
epochs = range(1, len(accuracy) + 1)

plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
plot = ax1.plot(epochs, accuracy, 'bo', label='Training Accuracy')
ax1.plot(epochs, valAccuracy, 'b', label='Validation Accuracy')
ax1.set(title='Training/Validation Accuracy', ylabel='Accuracy')
ax1.legend()

plot = ax2.plot(epochs, loss, 'bo', label='Training Loss')
ax2.plot(epochs, valLoss, 'b', label='Validation Loss')
ax2.set(title='Training/Validation Loss', ylabel='Loss', xlabel='Epochs')
ax2.legend()

fig.suptitle('Loss/Accuracy of the LSTM Sentiment Classifier',
             fontsize=16, fontweight='bold')


# In[ ]:


# The model still overfits: In the absence of more data, the overfitting being experienced can probably be
# minimized by reducing the number of layers or by reducing the number of units used in the neural
# architecture. The Dropout can also be increased. Weight regularization via the keras
# kernel_regularizer can also be implemented.


# In[ ]:


results = model.evaluate(XTest, yTest)


# In[ ]:


print(classification_report(np.argmax(yTest, 1), model.predict_classes(XTest)))


# In[ ]:


# A test accuracy score of 0.84 gives a much improved performance compared to the Logistic Regression
# algorithm. A normal slight drop from the 0.87 recorded during validation evaluation.


# In[ ]:


model.save('./LSTM classifier.h5')
keras.models.load_model('./LSTM classifier.h5')

