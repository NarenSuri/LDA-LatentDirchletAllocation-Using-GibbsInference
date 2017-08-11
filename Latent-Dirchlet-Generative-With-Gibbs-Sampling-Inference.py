# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 21:47:15 2016

@author: Naren Suri

"""
###############################################################################
# This code implements the Latent Dirchlet Allocation and Uses the 
# Geneative Approach to generate the sequence of words to define the topics
# the words belongs to. This method uses the Bag-Of-Words Approach
# Also, inorder to estimate the integral as shown in the below paper, we hav
# used the Gibbs Sampling Approach 

###############################################################################
## References :
# Latent Dirichlet Allocation David M. Blei , Andrew Y. Ng
# https://www.cs.princeton.edu/~blei/papers/BleiNgJordan2003.pdf

# Implementation of Latent Dirchlet Allocation using the below Researcch Papers
# http://u.cs.biu.ac.il/~89-680/darling-lda.pdf

# Bob Carpenter. Integrating out multinomial parameters in latent dirichlet
#allocation and naive bayes for collapsed gibbs sampling. Technical report,
#Lingpipe, Inc., 2010.
#https://lingpipe.files.wordpress.com/2010/07/lda3.pdf

###############################################################################

# using NumPy and Scipy and Gamma Functions
from scipy.special import gammaln # for gamma functions
import numpy as np # numoy operations
import scipy as sipy # for scientific python operations
import os
import shutil # for more easy file operations
#import nltk
import glob
from pathlib2 import Path
from unidecode import unidecode
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import random
import pprint

def CreateDataForLDA(Location,fileName=None):
    # Check if the folder that user wants to use already exists
    TotalFilesToProcess = 0
    DataHolder = []
    tokenizer = WordPunctTokenizer() 
    english_stops = set(stopwords.words('english')) 
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    if os.path.exists(Location):
        print "The Folder you specified already exists, all the files those you want to use should be here"
    else:
        os.mkdir(Location)
        print "Folder got created"

    for eachFileInFolder in glob.iglob(Location+"*.txt"):
        TotalFilesToProcess= TotalFilesToProcess + 1
        print "Now Processing the File : " + eachFileInFolder
        FileContents = Path(eachFileInFolder).read_text()
        tokens = tokenizer.tokenize(FileContents.encode('ascii',errors='ignore'))
        StopWordsRemovedText = [word for word in tokens if word not in english_stops]
        words=""
        for idx, word in enumerate(StopWordsRemovedText):
            words = words + " " +stemmer.stem(lemmatizer.lemmatize(word))
        DataHolder.append(words)
    print "Hello, basic stemming and stop word removal is done"
    return(DataHolder)




def generateVocab(DataHolder):
    set_unique_words=""
    
    for EachDocumentMessage in DataHolder:
        EachDocumentMessage.split()
        unique_words = set(EachDocumentMessage.split())
        set_unique_words = set_unique_words + ' '.join(unique_words)
        
    final_set_unique_words = set(set_unique_words.split())
    print "Created a list of all unique vocab and sent for usage"
    return(final_set_unique_words)

   
class MyLdaImplementation:
    # this is a generative approach for LDA-Gibbs Implementation

    def setAllParam(self, numberOfTopics=None, alpha=None, beta=None, GibbsIterations=None, MCMC_BURN_IN=None,
                     MCMC_THIN_INTERVAL=None, MCMC_SAMPLE_LAG=None,lengthOfVocab=None,vocabulary=None,DataHolder=None,dafaultDummyDocLenth=None):
            # Setting all parameters required for the MCMC sampling and the LDA implementation           
            
            self.V_lengthOfVocab = lengthOfVocab+1 # length of unique vocabulary
            
            self.dafaultDummyDocLenth = dafaultDummyDocLenth
            self.vocabulary = vocabulary   # unique vobulary         
            
            self.DataHolder = DataHolder # lists of messages
            
            self.M_TotalDocuments = len(DataHolder)
            
            self.alpha = alpha               # Setting the dirichlet parmaeter alpha , this is for Documents
            
            self.beta = beta             # Setting the dirichlet parmaeter Beta , this is for Topic and words
            
            self.K_topics = numberOfTopics         ## Config the number of topics it assumed
          
            ## Setting up everything for the gibbs sampler usage
            
            self.GibbsIterations = GibbsIterations             # max GibbsIterations
           
            self.MCMC_BURN_IN = MCMC_BURN_IN              # burn in period
            
            self.MCMC_THIN_INTERVAL = MCMC_THIN_INTERVAL             # thinning interval
            
            self.MCMC_SAMPLE_LAG = MCMC_SAMPLE_LAG             # sample lag
            

            
            
            print "self.MCMC_SAMPLE_LAG  " + str(self.MCMC_SAMPLE_LAG)
            print "All initial parameters you passed are Set"
            print "###############################  [ Calling the RandomDistributions_Initilizations_Latent_Documents_Topics]  ##########################"
            self.RandomDistributions_Initilizations_Latent_Documents_Topics()
            
   
   
   
    def GenerateDistributionOfWords(self):
       # generating a distribution of words 
       self.FillHalfOfTopicROws = self.K_topics /2
       self.Topic_Word_Distrib = np.zeros((self.K_topics, self.V_lengthOfVocab))
       
       for l in range(self.FillHalfOfTopicROws):
           self.Topic_Word_Distrib[l,:] = self.vertical_topic(l)
       
       for l in range(self.FillHalfOfTopicROws):
           self.Topic_Word_Distrib[l+self.FillHalfOfTopicROws,:] = self.vertical_topic(l)

       self.Topic_Word_Distrib/= self.Topic_Word_Distrib.sum(axis=1)[:, np.newaxis] # counts to  probabilities  
       Topic_Word_Distrib = self.Topic_Word_Distrib
       print "***** Finishedd Word Distributions Assignment***"

       return(self.Topic_Word_Distrib)
   
   
   
   
    def vertical_topic(self,topic_index):
        """
        Generating some random word doc distributions for each document vertically.
        """
        m = np.zeros(self.V_lengthOfVocab)
        randomIndexes = random.sample(range(self.V_lengthOfVocab),  self.V_lengthOfVocab/3)
        getSomeRandomNumber =  random.randint(self.dafaultDummyDocLenth/2, self.dafaultDummyDocLenth)
        for k in randomIndexes:
            m[k] = self.dafaultDummyDocLenth/getSomeRandomNumber
        return m         
         
    def ConvertDataHolderToIndexes(self):
        #for k in len(self.DataHolder):
        print "" 
            
            
    def RandomDistributions_Initilizations_Latent_Documents_Topics(self):
        # now lets assign all the latent variables of the graphial model with some rando initializations
        # these initializations are dependent on prior distributions
        # the two prior distributions i use here are Dirchlet and other Multinomial
        print "**********[ Prior Distributions assignment to the latent variables  ]***************"
        #self.ConvertDataHolderToIndexes(self)
        self.n__Words_Topics = np.zeros((self.V_lengthOfVocab, self.K_topics)) # total count of assignment of each word to a particular topic
        #print self.n__Word_Topics
        self.n__Documents_Topics = np.zeros((self.M_TotalDocuments, self.K_topics)) # Total count of assignment of each topic to a particular document
                
        self.total__words_InEach_Topic = [0] * self.K_topics # total number of words assigned to each topic .
        
        #######################################################################
        
        self.WordDistrib= self.GenerateDistributionOfWords()
        # after creating the Random topic- word distribution lets do the docuemnt word genration
        self.Docu_Word_Distrib = self.Document_Word_Distribution_Generation()
        print "Random asssignment of topics to words in the docuemnt"
        #print self.Docu_Word_Distrib
        
        #######################################################################
        
        #self.n__words_EachIn_Topic = np.sum(self.Docu_Word_Distrib, axis=1)
        #print self.n__words_EachIn_Topic
        
        # total number of words in eacch document
        self.total__words_InEach_Docu = [0] * self.M_TotalDocuments
        
        ######################################################################
        ## Here starts the implementation of the algorithm as shown in the rrsearch paper and as we showed in our report
        self.Topic_Assignment_EachWord_Z = []
        for m in range(self.M_TotalDocuments):
            N = len(self.DataHolder[m].split())
            self.Topic_Assignment_EachWord_Z.append([0]*N)
            for n in range(N-1):
                topic = int (random.random() * self.K_topics)
                self.Topic_Assignment_EachWord_Z[m][n] = topic
                print "updating the counts"
                print n
                print m
                print self.DataHolder[m].split()[n]
                
                indexToUse = self.vocabulary.index(self.DataHolder[m].split()[n])
                self.n__Words_Topics[indexToUse][topic] += 1
                self.n__Documents_Topics[m][topic]+= 1
                self.total__words_InEach_Topic[topic]+=1
            self.total__words_InEach_Docu = N 
            
        ## Gibbs and Bayes way to understand the nature of the iterations
        if self.SAMPLE_LAG > 0:

            self.theta_sum_val = np.zeros((self.M_TotalDocuments, self.K_topics))
            self.theta = np.zeros((self.M, self.K_topics))
            # cumulative statistics of phi
            self.phi_sum = np.zeros((self.K_topics, self.V))
            self.phi = np.zeros((self.K_topics, self.V))
            # size of statistics
            self.numstats = 0

      
        

    
    def generateEachDocument(self):
        # We have previously shown in the function calling this function on how the distribution
        # is being ependent and calleed between each other for sampling
     """
        This is the very important piece of the Generation of document with some distributional 
        assumptions:
            1) Topic proportions are to be sampled from the Dirichlet distribution.
            2) From the Multinomial Sample a topic index using the topic proportions from step 1).
            3) Sample a word from the Multinomial corresponding to the topic index from 2).
            4) Go to 2) if need another word.
        """          
        theta = np.random.mtrand.dirichlet([self.alpha] * self.K_topics)
        vocabFreqCount = np.zeros(self.V_lengthOfVocab)
        for n in range(self.dafaultDummyDocLenth):
            # Sample from MultiNomial Distribution of the data
            topic_index_z_sampled =  np.random.multinomial(1,theta).argmax()
            word_w_sampled_withz =  np.random.multinomial(1,self.Topic_Word_Distrib[topic_index_z_sampled,:]).argmax()
            vocabFreqCount[word_w_sampled_withz] += 1

        return vocabFreqCount        
      
      
      
      
    def Document_Word_Distribution_Generation(self):
        """
        This is the very important piece of the Generation of document with some distributional assumptions:
            1) Topic proportions are to be sampled from the Dirichlet distribution.
            2) From the Multinomial Sample a topic index using the topic proportions from step 1).
            3) Sample a word from the Multinomial corresponding to the topic index from 2).
            4) Go to 2) if need another word.
        """         
        
        Doc_Word_Distrib = np.zeros((self.M_TotalDocuments,  self.V_lengthOfVocab ))
        for i in xrange(self.M_TotalDocuments):
            Doc_Word_Distrib[i, :] = self.generateEachDocument()
        print "Doc Word Distrib generation"
        #print Doc_Word_Distrib
        return Doc_Word_Distrib   
        
    


        
###########################################################     
DataHolder = CreateDataForLDA(Location ='D:/sem3/TopicModeling/en/')
# generate Vocab Size and Vocab List
final_set_unique_words = generateVocab(DataHolder)
#print final_set_unique_words
vocabulary = list(final_set_unique_words)
#print vocabulary
lengthOfVocab = len(vocabulary)


## Working on LDA modeling
## call the LDA model here
documents = [
        [1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 6],
        [2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2],
        [1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 0],
        [5, 6, 6, 2, 3, 3, 6, 5, 6, 2, 2, 6, 5, 6, 6, 6, 0],
        [2, 2, 4, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 0],
        [5, 4, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2], ]

lda = MyLdaImplementation()

lda.setAllParam(numberOfTopics=16, alpha=5, beta=0.5, GibbsIterations=10000, MCMC_BURN_IN=2000,
                     MCMC_THIN_INTERVAL=100, MCMC_SAMPLE_LAG=10, lengthOfVocab=lengthOfVocab,vocabulary=vocabulary,DataHolder=DataHolder,dafaultDummyDocLenth=100) 
                     

       
        
    
    
    
 
    

