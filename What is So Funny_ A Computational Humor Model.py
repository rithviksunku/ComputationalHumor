#!/usr/bin/env python
# coding: utf-8

# ## What is So Funny: A Computational Humor Model
# ##### A Correlative Study Investigating Relationships between Joke Content and Level of Funniness

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
import random
import string
import sklearn

get_ipython().system('pip install TextBlob')
get_ipython().system('pip install vaderSentiment')
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier 
from sklearn.svm import SVC, LinearSVC, NuSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.classify import ClassifierI
from statistics import mode


# # Data Import

# In[2]:


#importing the jester ratings
dff = pd.read_csv('jester_ratings.csv')
dff.head()


# In[3]:


#importing the jester jokes
df = pd.read_csv('jester_items.csv')
df.head()


# # Data Exploration

# In[4]:


#example of a jester joke
df['jokeText'][0]


# In[5]:


#exploring the biases amongst individual user ratings
dff_test = dff.loc[dff['userId'] < 10]
dff_test[['userId','rating']].boxplot(by = 'userId', return_type = 'axes')
plt.title( 'Distribution of Ratings for Users 1 - 9' )
plt.ylabel('Joke Rating ( -10, 10 )', fontsize=10)
plt.xlabel('User ID #', fontsize=10)
plt.suptitle('')
plt.show()


# In[6]:


#exploring the distribution of joke ratings per joke
x = dff.loc[dff['jokeId'] == 7]['rating']
sn.distplot(x).set_title('Distribution of Ratings for Joke Id #7')
plt.ylabel('% Occurence', fontsize=10)
plt.xlabel('Joke Rating ( -10, 10 )', fontsize=10)
plt.suptitle('')
plt.show()


# # Data Cleaning

# In[7]:


#calculating top 5% rating for the joke depending on if it is considered funny or not funny
joke_rating = dff.groupby(['jokeId'])['rating'].agg(list).reset_index()
joke_rating['upper_rating'] = joke_rating['rating'].apply(lambda ratings: np.percentile(ratings, 95) if np.mean(ratings) > np.median(ratings) else np.percentile(ratings, 5))
joke_rating = joke_rating[['jokeId','upper_rating']]
joke_rating.head()


# In[8]:


#obtaining rating stats for each user to normalize scores --> will reduce bias amongst user
user = dff.groupby('userId')['rating'].agg(list).reset_index()
def getstats(row):
    ratings = row['rating']
    std_ratings = np.std(ratings)
    mean_ratings = np.mean(ratings)
    
    return { 'user_mean': mean_ratings,
            'user_std': std_ratings
    }
stats = pd.DataFrame(list(user.apply(getstats, axis = 1)))
user = pd.concat([user, stats], axis = 1)
user.head()


# In[9]:


#merging the user's normalized scores (calculated z-score) for each joke with the joke ids and user ids
dff = pd.merge(user, dff, on = 'userId')
dff['nom_rating'] = (dff['rating_y'] - dff['user_mean'] )/dff['user_std']
dff = dff[['userId','jokeId','nom_rating']]
dff = dff.dropna()
dff.head()


# In[10]:


#cleaning joke text to remove question and answer queues and sentence breaks
def remove_punctuations(text):
    for spaces in ['\w' ,'\s' ,'\n', 'Q.', 'A.','\t','Q:', 'A:']:
        text = text.replace(spaces, ' ')
    for no_spaces in ['\'']:
        text = text.replace(no_spaces, "")
    return text

df["jokeText"] = df['jokeText'].apply(remove_punctuations)


# In[11]:


#merging the two datasets to form one dataset with cleaned scores and jokes
jokes = pd.merge(df, dff, on='jokeId').reset_index()
jokes.head()
agg_rating = jokes.groupby('jokeId')['nom_rating'].agg(list).reset_index()
jokes = pd.merge(agg_rating, df, on = 'jokeId')
jokes = pd.concat([jokes, joke_rating['upper_rating']], axis = 1)
jokes.head()


# # Data Engineering

# In[12]:


#performing text analysis to gather statistics about jokes from their sentiment to the number of nouns
def analyze_joke(row):
    joke = row['jokeText']
    tokens = sent_tokenize(joke)
    num_sentences = len(tokens)
    word = TextBlob(joke)
    num_propnouns = sum(['JJ' in tag for _, tag in word.tags]) + sum(['NNP' in tag for _, tag in word.tags])
    num_nouns = sum(['NN' in tag for _, tag in word.tags])
    num_words = len(word.words)
    num_characters = len(joke)
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(joke)
    
    rating = row['nom_rating']
    ratings = row['upper_rating']
    max_rating = max(rating)
    min_rating = min(rating)
    
    funny = 1
    if ratings < 0:
        funny = 0
    
    return {'polarity': word.sentiment.polarity,
            'subjectivity': word.sentiment.subjectivity,
            'neg_sentiment':sentiment['pos'],
            'pos_sentiment':sentiment['neg'],
            'compound_sentiment':sentiment['compound'],
            'num_nouns': num_nouns,
            'num_propnouns': num_propnouns,
            'ARI':(4.71 * (num_characters/num_words) + 0.5 * (num_words /num_sentences) - 21.43),
            'num_punctuations': len(word.tokenize()) - num_words,
            'max_rating': max_rating,
            'min_rating': min_rating,
            'average_score':np.mean(row['nom_rating']),
            'median_score':np.mean(row['nom_rating']),
            'noun_percent': ((num_propnouns + num_nouns) / num_words),
            'funny': funny
           }


# In[13]:


#merging statistics to original joke dataset
analysis = pd.DataFrame(list(jokes.apply(analyze_joke, axis = 1)))
jokes = pd.concat([jokes, analysis], axis = 1)
jokes.head()


# # Data Visualization

# In[14]:


#visualizing the sentiment of the joke and its effect on the funniness noted by 1 for funny and 0 for not funny
y = jokes['subjectivity']
z = jokes['polarity']
n = jokes['funny']

fig, ax = plt.subplots()
ax.scatter(z, y, color = 'red')

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))
    
plt.scatter(jokes['polarity'],jokes['subjectivity'], color = 'red')
plt.title('Sentiment Analysis', fontsize = 20)
plt.xlabel('← Negative — — — — — — Positive →', fontsize=10)
plt.ylabel('← Facts — — — — — — — Opinions →', fontsize=10)


# In[15]:


#visualizing the relationship of number of nouns on the overall score of the joke (average z-score for joke)
yaxis = [jokes['average_score'], jokes['upper_rating']]
actual_values =  yaxis[0]
metric = jokes['num_nouns']

z = actual_values
y = metric

fig, ax = plt.subplots()
ax.scatter(y, z, color = 'blue')
plt.title('Number of Nouns vs Joke Ratings', fontsize = 20)
plt.xlabel('metric', fontsize=10)
plt.ylabel('actual scores', fontsize=10)


# # Data Modeling

# ## 1. Mutlivariate Regression Model
# I first performed a multivariate regression to see the impact of various statistics on the ratings of the jokes

# In[16]:


#split data in to training and testing sets 
train = jokes[:98]
test = jokes[98:]


# In[17]:


#fitting data around the statistics and creating linear regression model
variables = ['ARI', 'compound_sentiment','neg_sentiment', 'noun_percent', 'num_nouns',
       'num_propnouns', 'num_punctuations', 'polarity', 'pos_sentiment',
       'subjectivity']
num_vars = len(variables)

reg = linear_model.LinearRegression()
reg.fit(train[variables],train.average_score)


# In[18]:


#evaluating the individual parameters to see which ones had the greatest effect
reg.coef_


# In[19]:


#evaluating performance of model using rsquared and adjusted rsquared
rsquare = reg.score(train[variables], train.average_score)
r = np.sqrt(rsquare)
adjusted_rsquare = 1 - (((1-rsquare)*(len(train) - 1))/(len(train) - num_vars - 1))
mse = sklearn.metrics.mean_squared_error(train.average_score, reg.predict(train[variables]))
rmse = np.sqrt(mse)
print('R^2:', rsquare, 'r:', r, 'adjusted R^2:', adjusted_rsquare, 'rmse:', rmse)


# #### Evaluating Relationship between Test Statistics and Ratings:
# 
# - 'ARI' --> (weak, linear, positive) 
# 
# - 'compound_sentiment' --> (no difference)
# 
# - 'neg_sentiment' --> (medium, linear, negative)
# 
# - 'noun_percent' --> (weak, linear, negative)
# 
# - 'num_nouns' -->  (strong, logarithmic, positive)
# 
# - 'num_propnouns' -->  (medium, logarithmic, positive)
# 
# - 'num_punctuations' --> (medium, logarithmic, positive)
# 
# - 'polarity' --> (weak, linear, negative)
# 
# - 'pos_sentiment' --> (weak, linear, negative)
# 
# - 'subjectivity' --> (weak, linear, negative)

# ## 2. Text Classification 
# I next performed a various classification models to see based on content whether a text is funny or not funny

# In[20]:


#creating the classification text set which has tokenized words from the joke and if it is funny or not (denoted as 1 or 0)
classification_set = jokes[['jokeText','funny']]
classification_set = classification_set[['jokeText','funny']]
classification_set['jokeText'] = classification_set['jokeText'].apply(word_tokenize)
classification_set['tuples'] = classification_set.apply((lambda x: (x['jokeText'], x['funny'])), axis = 1)
classification_set.head()


# In[21]:


#cleaned it up to put it in the necessary format 
tups = classification_set['tuples']
classification_cleaned = [tups[i] for i in range(len(tups))]
random.shuffle(classification_cleaned)


# In[22]:


#gathered all words from the joke texts and removed punctuation and made every word lowercase
all_words = []
stop_words = set(stopwords.words('english')) 

for lists in classification_set['jokeText']:
    for words in lists:
        all_words.append(words.lower())
        
all_words = [w for w in all_words if not w in string.punctuation and not w in "``'''..." and not w in stop_words]


# In[23]:


#Explored which words are most common
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(10))


# In[24]:


#created word feature set which will be used in text classification and whether the word was considered funny or not
word_features = list(all_words.keys())[:4000]
def find_features(classification_cleaned):
    words = set(classification_cleaned)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
featuresets = [(find_features(joke), funny) for (joke, funny) in classification_cleaned]


# In[31]:


#creating testing and training set to train classification model
training_set = featuresets[:98]
testing_set = featuresets[98:]
training_set


# In[26]:


#performed Naive Bayes classification and showed what are the most informative words
classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Naive Bayes Classifier Accuracy %:', (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(5)


# In[27]:


#created a class to find best classifier and show percent confidence to which all classifers chose the same output
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# In[28]:


#created various other classifiers to include in the voting process

#Bernoulli Classifier
Bclassifier = SklearnClassifier(BernoulliNB())
Bclassifier.train(training_set)
print('Bernoulli Classifier Accuracy %:',(nltk.classify.accuracy(Bclassifier, testing_set))*100)

#Logistic Regression Classifier
LogisticRegressionclassifier = SklearnClassifier(LogisticRegression())
LogisticRegressionclassifier.train(training_set)
print('Logistic Regression Classifier Accuracy %:',(nltk.classify.accuracy(LogisticRegressionclassifier, testing_set))*100)

#Stochastic Gradient Descent Classifier
SGDClassifierclassifier = SklearnClassifier(SGDClassifier())
SGDClassifierclassifier.train(training_set)
print('Stochastic Gradient Descent Classifier Accuracy %:',(nltk.classify.accuracy(SGDClassifierclassifier, testing_set))*100)

#Support Vector Machine Classifier
SVCclassifier = SklearnClassifier(SVC())
SVCclassifier.train(training_set)
print('SVM Classifier Accuracy %:',(nltk.classify.accuracy(SVCclassifier, testing_set))*100)

#Linear Support Vector Machine Classifier
LinearSVCclassifier = SklearnClassifier(LinearSVC())
LinearSVCclassifier.train(training_set)
print('Linear SVM Classifier Accuracy %:',(nltk.classify.accuracy(LinearSVCclassifier, testing_set))*100)


# In[29]:


#implemented voted_classifier which gets the accuracy
voted_classifier = VoteClassifier(classifier,Bclassifier,LogisticRegressionclassifier,SGDClassifierclassifier,SVCclassifier,LinearSVCclassifier)
print('Voted Classifier Accuracy %:',(nltk.classify.accuracy(voted_classifier, testing_set))*100)


# In[30]:


#model tested on one joke to show confidence in all models
testing = testing_set[13][0]
print('Classification:', voted_classifier.classify(testing), "Confidence %: ", voted_classifier.confidence(testing)*100)


# ### Conclusion
# Through this analysis, we determined that although there might be other factors at play, understanding what makes a joke funy by its content seems to be accurate (classifier) and therefore can stand alone (linear regression) in predicting what makes something funny beyond the common notions of delivery, enivironment, pyschological effects at play.
