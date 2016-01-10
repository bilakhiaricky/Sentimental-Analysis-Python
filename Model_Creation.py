"""
Created on Sun Nov 22 17:26:01 2015
Script will do sentiment analysis on restaurant reviews.
@author: Ricky
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.util import ngrams
import re
from sklearn import linear_model

fileWriter = open('out.txt','w')

mystopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

def loadData(fname):
    reviews=[]
    labels=[]
   # count2 = 0
    f=open(fname)
    for line in f:
       # count2 = count2 + 1
        review,rating=line.strip().split('\t')
        review = re.sub('not ', 'not', review)
        review = re.sub('Not ', 'Not', review)
        review = re.sub('<br>', ' ',review)
        review = re.sub(' +', ' ',review)
       # review = re.sub('[^a-z\d]', ' ',review)
        terms = review.split()
        reviews.append(review.lower())    
        labels.append(int(rating))
    threegrams = ngrams(terms,3)
    for tg in threegrams:
        if tg[0] in mystopwords or tg[1] in mystopwords or tg[2] in mystopwords:
            continue
  #  print count2  
    f.close()
    return reviews,labels

def loadTrainData(fname):
    reviews=[]
    f=open(fname)
    for line in f:
        review=line.strip()
        review = re.sub('not ', 'not', review)
        review = re.sub('Not ', 'Not', review)
        review = re.sub('<br>', ' ',review)
        review = re.sub(' +', ' ',review)
       # review = re.sub('[^a-z\d]', ' ',review)
        terms = review.split()
        reviews.append(review.lower())
    threegrams = ngrams(terms,3)
    for tg in threegrams:
        if tg[0] in mystopwords or tg[1] in mystopwords or tg[2] in mystopwords:
            continue
    f.close()
    return reviews

rev_train,labels_train=loadData('training.txt')
rev_test=loadTrainData('testing.txt')


MNB_pipeline = Pipeline([('vect', CountVectorizer(ngram_range = (1, 2))), 
                         ('clf', MultinomialNB(alpha = 1.0, fit_prior = True)),
                        ])

KNN_pipeline = Pipeline([('vect', CountVectorizer()), 
                         ('clf', KNeighborsClassifier(n_neighbors = 20)),
                        ])
                        
SGD_pipeline = Pipeline([('vect', CountVectorizer()),
                        ('clf', linear_model.SGDClassifier(loss='log')),
                        ])
                        
LR_pipeline = Pipeline([('vect', CountVectorizer()), 
                        ('tfidf', TfidfTransformer(norm = 'l2', use_idf = True, smooth_idf = True, sublinear_tf = True)),
                        ('clf', LogisticRegression(warm_start = True, random_state = 1)),
                       ]) 
                     

eclf = VotingClassifier(estimators=[('MNB', MNB_pipeline), ('SGD',SGD_pipeline), ('LR', LR_pipeline)], voting = 'soft', weights = [3,2,3])
#('KNN', KNN_pipeline), 

eclf.fit(rev_train,labels_train)

#use soft voting to predict (majority voting)
pred=eclf.predict(rev_test)

for x in pred:
    fileWriter.write(str(x)+'\n')
fileWriter.close()
