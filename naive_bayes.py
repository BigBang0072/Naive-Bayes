import numpy as np
import pandas as pd

#Class Definition for the Naive Bayes Classifier
class NaiveBayes():
    #Training Attributes: for probability Estimation
    train_word_bag=None
    train_dataset=None
    total_pos=None
    total_neg=None

    #Validation/Testing Attributes
    test_dataset=None

    #Initialization function
    def __init__(self,dataset):
        '''
        DESCRIPTION:
            The constructor for this class.
        '''
        #initializing the training datset
        self.total_pos=dataset[0]
        self.total_neg=dataset[1]
        self.train_dataset=dataset[2]
        self.train_word_bag=dataset[3]

        #Retreiving the validation datset
        self.test_dataset=dataset[4]

    #Function to learning the probability of each word in respective world
    def estimate_word_prob(self):
        '''
        1. TRAIN time:
        This function will estimate the probability of occurance of
        a word in the world of a particular sentiment using the maximum
        likeliehood estimate as we did for the bernoulli trial.

        2. TEST time:
        The estimate will be done only for the data in training set and
        will not contain the words which might not be in training set.
        So, we will skip those words from the review assuming they are
        only specific to that particular class hence having probability
        in that world as 1.
        '''
        
