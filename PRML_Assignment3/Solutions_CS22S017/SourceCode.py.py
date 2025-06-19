#!/usr/bin/env python
# coding: utf-8

# # PRML ASSIGNMENT 3: SPAM/HAM
# #(Request before running code:test folder is kept empty as no test datafile is given. Please add files to this folder before running below code)

# Training Dataset spam_ham_dataset downloaded from
# https://www.kaggle.com/datasets/venky73/spam-mails-dataset?resource=download

# In[95]:


import numpy as np
import sys
import os #to list all files in test directory

#for preprocessing,vocabBuild
from collections import UserList
from fileinput import filename
import math
import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words

#for vectorizer
import ast

#for naive Bayes
from sklearn.model_selection import train_test_split
import pickle


# Preprocess test/training data:
# Remove duplicate data, Convert data to lower case, Remove special characters and stopwords(eg:the, it..),stemming 

# In[96]:


class ForPreprocessing:

    def __init__(self, trainDataset, training):
        self.training = training
        self.trainDataset = trainDataset
        self.processedTrainDataset = None

    # remove duplicate values and convert text to lower case
    def removeDup_makeLower(self):
        if(self.training):
            temp = pd.read_csv(self.trainDataset , usecols=['text', 'spam'])
            temp.rename(columns={'spam':'label_num'}, inplace=True)
            temp['class'] = np.where(temp['label_num'] == 1, 'spam', 'ham')
        else:
            temp = pd.read_csv(self.trainDataset, usecols=['text'])
        temp.drop_duplicates(inplace=True)
        self.processedTrainDataset = temp
        self.processedTrainDataset['text'] = self.processedTrainDataset['text'].apply(lambda x : x.lower()) #text to lowercase
    
    # remove special char and stopwords
    def removeSpCharStopword(self):
        spChar = "!#$%&'()*+,-./:;<=>?@[\]\\^_`{|}~'"
        num = '0-9'
        self.processedTrainDataset['text'] = self.processedTrainDataset['text'].apply(lambda x : re.sub(r'[{0}]'.format(spChar + '"') , '', x))
        self.processedTrainDataset['text'] = self.processedTrainDataset['text'].apply(lambda x : re.sub(r'[{0}]'.format(num), '', x))
        self.processedTrainDataset["text"] = self.processedTrainDataset["text"].apply(lambda x: re.sub(' +', ' ', x))
    
        sw = set(stopwords.words('english'))
        sw.add('http')
        sw.add('subject')
        self.processedTrainDataset['text'] = self.processedTrainDataset['text'].apply(lambda x : " ".join([word for word in str(x).split() if word not in sw]))
    
    # reducing the words to its base form using stemming or lemmatization. Inflection
    def inflection(self, st_lm):
        if(st_lm == 1):
            ps = PorterStemmer()
            self.processedTrainDataset['text'] = self.processedTrainDataset['text'].apply(lambda x : " ".join([ps.stem(word) for word in x.split()]))
        else:
            lm = WordNetLemmatizer()
            self.processedTrainDataset['text'] = self.processedTrainDataset['text'].apply(lambda x : " ".join([lm.lemmatize(word) for word in x.split()]))

    # keeping the words present in dictionary 
    def keepDictWords(self):
        words2 = set(words.words())
        self.processedTrainDataset['text'] = self.processedTrainDataset['text'].apply(lambda x : " ".join([word for word in x.split() if word in words2]))    

    # saving the preprocessed data as a new file
    def saveProcessedTrainDataset(self, fileName):
        self.processedTrainDataset.to_csv(fileName + '.csv', index=False)


# Build the test dataset

# In[97]:


class TestSet:

    def __init__(self, TestDataPath):
        self.TestDataPath = TestDataPath
        self.pathToSave = 'datasets/testEmails.csv'
    
    # creating a csv file for the test emails in the test folder
    def makeTestCsv(self):
        self.emails = []
        for count, sub_files in enumerate(os.listdir(self.TestDataPath)):
            file_path = self.TestDataPath + '/' + sub_files
            fp = open(file_path, 'r')
            self.emails.append(fp.read())
            fp.close()
    
    # saving the csv file
    def saveAsCsv(self):
        dataToSave = {'text' : self.emails}
        df = pd.DataFrame(dataToSave)
        df.to_csv(self.pathToSave, index=False)
    
    # using the preprocessor class to clean the test text data
    def cleanAndSave(self):
        preObj = ForPreprocessing(self.pathToSave, 0)
        preObj.removeDup_makeLower()
        preObj.removeSpCharStopword()
        preObj.inflection(0)
        preObj.keepDictWords()
        preObj.saveProcessedTrainDataset('testEmails')
        


# Vocabulary Building: make the vocabulary for all the words in the organized training dataset

# In[98]:


class BuildVocab:

    def __init__(self, ipFilepath):
        self.ipFilepath = ipFilepath
        self.ipData = pd.read_csv(ipFilepath)
        self.wordsDict = {}
        self.invFreq = {}
        self.words2 = set(words.words())
    
    # build and save as text file the word vocabulary.
    def buildWordDict(self, filename):
        for x in range(self.ipData.shape[0]):
            try:
                email = self.ipData.loc[x, 'text'].split() 
                self.addToDict(email)
                self.invDocFreq(email)
            except:
                self.ipData.drop(x, inplace=True)
                continue
        fp1 = open(filename + '.txt', 'w')
        fp2 = open('invertedInd.txt', 'w')
        fp1.write(str(self.wordsDict))
        fp2.write(str(self.invFreq))
        fp1.close()
        fp2.close()
        self.ipData.to_csv(self.ipFilepath, index=False)
    
    # adding the words to the dictionary for each email
    def addToDict(self, email):
        index = len(self.wordsDict)
        for word in email:
            if word not in self.wordsDict:
                self.wordsDict[word] = index
                index += 1
    
    # creating the inverse document frequency table
    def invDocFreq(self, email):
        emailSet = set(email)
        for words in emailSet:
            if words not in self.invFreq:
                self.invFreq[words] = 1
            else:
                self.invFreq[words] += 1 
    


# Vectorize : Extract features from data

# In[99]:


class Vectorize:

    def __init__(self, dataPath, wordsDataPath, train2):
        self.dataset = pd.read_csv(dataPath)
        self.wordsData = ast.literal_eval(open(wordsDataPath, 'r').read())
        self.invInd = ast.literal_eval(open('invertedInd.txt', 'r').read())
        self.dataX = np.zeros((self.dataset.shape[0], len(self.wordsData)))
        self.invIndTable = np.zeros((self.dataset.shape[0], len(self.wordsData)))
        self.train2 = train2
        if(self.train2):
            self.dataY = np.zeros((self.dataset.shape[0]))

    # creating vector representation for each email in the training or testing data
    def vectorize(self, invInd):
        for x in range(self.dataset.shape[0]):
            try:
                email = self.dataset.loc[x, 'text'].split()
                for word in email:
                    if word in self.wordsData:
                        self.dataX[x, self.wordsData[word]] += 1
                        self.invIndTable[x, self.wordsData[word]] = self.invInd[word]
            except:
                continue
            E1 = np.log(self.dataset.shape[0] * np.ones(self.dataX[x , :].shape))
            E2 = np.log(self.invIndTable[x, :], out=np.zeros_like(self.invIndTable[x, :]), where=(self.invIndTable[x, :] != 0))
            self.invIndTable[x, :] = E1 - E2 #subtract second expression from first expression
            if(invInd):
                self.dataX[x, :] = self.dataX[x, :] * self.invIndTable[x, :]
            if(self.train2):
                self.dataY[x] = self.dataset.loc[x, 'label_num']
    
    # save the data for training and testing as npy binary files
    def saveData(self):
        if(self.train2):
            np.save('Xtrain.npy', self.dataX)
            np.save('Ytrain.npy', self.dataY)
        else:
            np.save('Xtest.npy', self.dataX)


# Naive Bayes Classifier with Gaussian likelihood assumption

# In[100]:


class NBC:

    def __init__(self, dataX, train2, nClasses=2, datay=None):
        self.train2 = train2
        self.X = dataX
        self.means = {}
        self.variances = {}
        self.classes = nClasses
        if(self.train2):
            if(datay is not None):
                self.y = datay
                self.classes = len(np.unique(datay))
        self.noData, self.features = dataX.shape
        self.eta = 0.00001
        self.priors = {} 

    # find and save in npy file; mean and variance parameters for two classes
    def findParameters(self):
        for x in range(self.classes):
            xClassC = self.X[self.y == x]
            self.means[str(x)] = np.mean(xClassC, axis=0)
            self.variances[str(x)] = np.var(xClassC, axis=0)
            self.priors[str(x)] = xClassC.shape[0] / self.noData
        np.save('means.npy', self.means)
        np.save('variances.npy', self.variances)
    
    # load npy files for the parameters to classify test data
    def loadParameters(self):
        means = np.load('means.npy', allow_pickle=True)
        variances = np.load('variances.npy', allow_pickle=True)
        for x in range(self.classes):
            self.priors[str(x)] = 1 / self.noData
            self.means[str(x)] = means[()][str(x)]
            self.variances[str(x)] = variances[()][str(x)]

    # claulating the posterior using Bayes rule
    def posterior2(self):
        likelihoods = np.zeros((self.noData, self.classes))
        for x in range(self.classes):
            likelihoodClassC = self.logLikelihood(self.means[str(x)], self.variances[str(x)])
            likelihoods[:, x] = likelihoodClassC + np.log(self.priors[str(x)])
        self.likelyClass = np.argmax(likelihoods, axis=1)

    # estimating the log likelihood for the data using multivariate gaussian density function
    def logLikelihood(self, mu, sigma):
        t1 = (-0.5 * np.sum(np.log(np.sqrt(sigma + self.eta)))) - (self.features * 0.5 * np.log(2 * math.pi))
        diff = np.zeros(self.X.shape)
        diff = (self.X - mu)
        t2 = -1 * np.sum(np.power(diff, 2)/(0.5 * (sigma + self.eta)), axis=1)
        return t1 + t2 #Return sum of first and second terms


# Training procedure

# In[101]:


def training(idfValue):
    
    #preprocess the training data
    preprocessObj = ForPreprocessing('datasets/spam_ham_dataset.csv', 1) #Absolute pathname. Make sure training dataset is present in same folder
    preprocessObj.removeDup_makeLower()
    preprocessObj.removeSpCharStopword()
    preprocessObj.inflection(0)    
    preprocessObj.saveProcessedTrainDataset('trainedEmails') 
    
    #make the vocabulary for all the words in the organized training dataset
    vocab = BuildVocab('trainedEmails.csv')
    vocab.buildWordDict('wordsDict')
    
    #Extract features from data
    dataPath = 'trainedEmails.csv' # 'test_emails.csv'
    wordsDataPath = 'wordsDict.txt'
    vec = Vectorize(dataPath, wordsDataPath, 1)
    vec.vectorize(idfValue)
    vec.saveData()
    
    #Output train accuracy of Naive bayes
    X = np.load('Xtrain.npy')
    Y = np.load('Ytrain.npy')
    bayes = NBC(X , 1, datay=Y)    
    bayes.findParameters()
    bayes.posterior2()
    print("Training data Accuracy: ")
    print(sum(bayes.likelyClass == Y)/X.shape[0])        


# Testing procedure

# In[102]:


def testing(idfValue = 0):

    training(idfValue)
    
    #build the test dataset
    testSet = TestSet('test')
    testSet.makeTestCsv()
    testSet.saveAsCsv()
    testSet.cleanAndSave()

    #extract features from the data
    dataPath = 'testEmails.csv'
    wordsDataPath = 'wordsDict.txt'
    vec = Vectorize(dataPath, wordsDataPath, 0)
    vec.vectorize(idfValue)
    vec.saveData()

    #run the naive bayes algorithm
    X = np.load('Xtest.npy')
    bayes = NBC(X, 0)
    bayes.loadParameters()
    bayes.posterior2()
    print('Predictions for the emails in test folder : ')
    print(bayes.likelyClass)


# Main function

# In[103]:


idfValue = 0 #0-without idf values 1-with idf value @argv
if len(sys.argv) > 1:
    #idfValue = int(sys.argv[1])
    idfValue = 1
    
testing(idfValue) #Fn call to run training procedure

    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




